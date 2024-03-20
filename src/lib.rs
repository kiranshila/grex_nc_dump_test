use color_eyre::eyre;
use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

const CHANNELS: usize = 2048;

/// Payload as they come from the NIC
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Payload {
    count: u64,
    pol_a: [Complex<i8>; CHANNELS],
    pol_b: [Complex<i8>; CHANNELS],
}

impl Payload {
    /// Yields an [`ndarray::ArrayView3`] of dimensions (Polarization, Channel, Real/Imaginary)
    fn as_ndarray_data_view(&self) -> ArrayView3<i8> {
        // C-array format, so the pol_a, pol_b chunk is in memory as
        //        POL A               POL B
        //  CH1   CH2   CH3  ...  CH1   CH2   CH3
        // [R I] [R I] [R I] ... [R I] [R I] [R 1]
        // Which implies a tensor with dimensions Pols (2), Chan (2048), Reim (2)
        // As the first index is the slowest changing in row-major (C) languages
        let raw_ptr = self.pol_a.as_ptr();
        // Safety:
        // - The elements seen by moving ptr live as long 'self and are not mutably aliased
        // - The result of ptr.add() is non-null and aligned
        // - It is safe to .offset() the pointer repeatedely along all axes (it's all bytes)
        // - The stides are non-negative
        // - The product of the non-zero axis lenghts (2*CHANNELS*2) does not exceed isize::MAX
        unsafe { ArrayView::from_shape_ptr((2, CHANNELS, 2), std::mem::transmute(raw_ptr)) }
    }

    pub fn random() -> Self {
        let mut pol_a = [Default::default(); CHANNELS];
        let mut pol_b = [Default::default(); CHANNELS];

        pol_a.par_iter_mut().for_each(|x| *x = rand::random());
        pol_b.par_iter_mut().for_each(|x| *x = rand::random());

        Self {
            count: Default::default(),
            pol_a,
            pol_b,
        }
    }
}

/// The voltage dump ringbuffer
#[derive(Debug)]
pub struct DumpRing {
    /// The next time index we write into
    write_ptr: usize,
    /// The data itself (heap allocated)
    buffer: Array4<i8>,
    /// The number of time samples in this array
    capacity: usize,
    /// The timestamp (packet count) of the oldest sample (pointed to by read_ptr).
    /// None if the buffer is empty
    oldest: Option<u64>,
    // If the buffer is completly full
    full: bool,
}

impl DumpRing {
    pub fn new(capacity: usize) -> Self {
        // Allocate all the memory for the array
        let buffer = Array::zeros((capacity, 2, CHANNELS, 2));
        Self {
            buffer,
            capacity,
            write_ptr: 0,
            full: false,
            oldest: None,
        }
    }

    pub fn push(&mut self, pl: &Payload) {
        // Copy the data into the slice pointed to by the write_ptr
        let data_view = pl.as_ndarray_data_view();
        self.buffer
            .slice_mut(s![self.write_ptr, .., .., ..])
            .assign(&data_view);

        // Move the pointer
        self.write_ptr = (self.write_ptr + 1) % self.capacity;
        // If there was no data update the timeslot of the oldest data and increment the write_ptr
        if self.oldest.is_none() {
            self.oldest = Some(pl.count);
            // Nothing left to do
            return;
        }

        // If we're full, we overwrite old data
        // which increments the payload count of old data by one
        // as they are always monotonically increasing by one
        if self.full {
            self.oldest = Some(self.oldest.unwrap() + 1);
        }

        // If we wrapped around the first time, we are now full
        if self.write_ptr == 0 && !self.full {
            self.full = true;
        }
    }

    /// Get the two array views that represent the time-ordered, consecutive memory chunks of the ringbuffer.
    /// The first view will always have data in it, and the second view will be buffer_capacity - length(first_view)
    fn consecutive_views(&self) -> (ArrayView4<i8>, ArrayView4<i8>) {
        // There are four different cases
        // 1. the buffer is empty or
        // 2. The buffer has yet to be filled to capacity  (and we always start at index 0) so there's only really one chunk
        if !self.full {
            (
                self.buffer.slice(s![..self.write_ptr, .., .., ..]),
                ArrayView4::from_shape((0, 2, CHANNELS, 2), &[]).unwrap(),
            )
        } else {
            // 3. The buffer is full and the write_ptr is at 0 (so the buffer is in order) or
            // 4. The write_ptr is non zero and the buffer is full, meaning the write_ptr is the split where data at its value to the end is the oldest chunk
            (
                self.buffer.slice(s![self.write_ptr.., .., .., ..]),
                self.buffer.slice(s![..self.write_ptr, .., .., ..]),
            )
        }
    }

    pub fn dump(&self, chunk_size: usize) -> eyre::Result<()> {
        // Create a tmpfile for this dump, as that will be on the OS drive (probably),
        // which should be faster storage than the result path
        let tmp_path = std::env::temp_dir();
        let tmp_file_path = tmp_path.join("test.nc");
        let mut file = netcdf::create(tmp_file_path)?;

        // Add the file dimensions
        file.add_dimension("time", self.capacity)?;
        file.add_dimension("pol", 2)?;
        file.add_dimension("freq", CHANNELS)?;
        file.add_dimension("reim", 2)?;

        // Describe the dimensions
        let mut mjd = file.add_variable::<f64>("time", &["time"])?;
        mjd.put_attribute("units", "Days")?;
        mjd.put_attribute("long_name", "TAI days since the MJD Epoch")?;

        let mut pol = file.add_string_variable("pol", &["pol"])?;
        pol.put_attribute("long_name", "Polarization")?;
        pol.put_string("a", 0)?;
        pol.put_string("b", 1)?;

        let mut freq = file.add_variable::<f64>("freq", &["freq"])?;
        freq.put_attribute("units", "Megahertz")?;
        freq.put_attribute("long_name", "Frequency")?;

        let mut reim = file.add_string_variable("reim", &["reim"])?;
        reim.put_attribute("long_name", "Complex")?;
        reim.put_string("real", 0)?;
        reim.put_string("imaginary", 1)?;

        // Setup our data block
        let mut voltages = file.add_variable::<i8>("voltages", &["time", "pol", "freq", "reim"])?;
        voltages.put_attribute("long_name", "Channelized Voltages")?;
        voltages.put_attribute("units", "Volts")?;

        // Write to the file, one timestep at a time (chunking in pols, channels, and reim)
        // We want chunk sizes of 64MB, which works out to 16384 time samples
        voltages.set_chunking(&[chunk_size, 2, CHANNELS, 2])?;
        //voltages.set_compression(0, true)?;

        let (a, b) = self.consecutive_views();
        let a_len = a.len_of(Axis(0));
        voltages.put((..a_len, .., .., ..), a)?;
        voltages.put((a_len.., .., .., ..), b)?;

        Ok(())
    }
}
