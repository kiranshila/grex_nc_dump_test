use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use grex_nc_dump_test::{DumpRing, Payload};

fn dump(c: &mut Criterion) {
    let n = 2usize.pow(20);
    let mut dr = DumpRing::new(n);
    for _ in 0..n {
        dr.push(&Payload::random());
    }
    let mut group = c.benchmark_group("dump");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);
    // Chunk sizes from 0.5 to 64 MiB
    // 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64 MiB
    for size in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                dr.dump(size).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, dump);
criterion_main!(benches);
