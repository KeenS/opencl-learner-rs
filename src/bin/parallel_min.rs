use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::error_codes::cl_int;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{
    Buffer, CL_MAP_READ, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE,
    CL_MEM_WRITE_ONLY,
};
use opencl3::program::Program;
use opencl3::types::{cl_uint, CL_TRUE};
use opencl3::Result;
use std::mem;
use std::ptr;

static SOURCE: &str = include_str!("../parallel_min.cl");

fn main() -> Result<()> {
    const NUM_SRC_ITEMS: usize = 4096 * 4096;
    let mut src_ptr = Vec::<cl_uint>::with_capacity(NUM_SRC_ITEMS);
    src_ptr.resize(NUM_SRC_ITEMS, 0);
    let a: cl_uint = 21341;
    let mut b: cl_uint = 23458;
    let mut min = cl_uint::MAX;
    for cell in &mut src_ptr {
        b = a.wrapping_mul(b & 65535);
        *cell = b + (b >> 16);
        min = if *cell < min { *cell } else { min };
    }
    println!("{min}");

    let dev = CL_DEVICE_TYPE_GPU as cl_uint;
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    let compute_units = device.max_compute_units().expect("max compute_units") as usize;
    let ws = 64usize;
    // 7 wavefronts per SIMD
    let mut global_work_size: usize = compute_units * 7 * ws;
    while ((NUM_SRC_ITEMS) / 4) % global_work_size != 0 {
        global_work_size += ws;
    }
    let local_work_size = ws;
    let num_groups = global_work_size / local_work_size;

    let context = Context::from_device(&device).expect("Context::from_device failed");
    let queue = CommandQueue::create(&context, context.default_device(), 0)
        .expect("CommandQueue::create failed");
    let program = Program::create_and_build_from_source(&context, SOURCE, "-cl-std=CL2.0")
        .expect("Program::create_and_build_from_source failed");
    let minp = Kernel::create(&program, "minp").expect("Kernel::create failed");
    let reduce = Kernel::create(&program, "reduce").expect("Kernel::create failed");

    ////////////////////////////////////////////////////////////////
    // Compute data

    let src_buffer = Buffer::<cl_uint>::create(
        &context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        NUM_SRC_ITEMS,
        src_ptr.as_mut_ptr() as *mut _,
    )?;
    let dst_buffer =
        Buffer::<cl_uint>::create(&context, CL_MEM_READ_WRITE, num_groups, ptr::null_mut())?;
    let dbg_buffer = Buffer::<cl_uint>::create(
        &context,
        CL_MEM_WRITE_ONLY,
        global_work_size,
        ptr::null_mut(),
    )?;
    let num_src_items = NUM_SRC_ITEMS as cl_int;
    let minp_event = ExecuteKernel::new(&minp)
        .set_arg(&src_buffer)
        .set_arg(&dst_buffer)
        .set_arg_local_buffer(mem::size_of::<cl_uint>())
        .set_arg(&dbg_buffer)
        .set_arg(&num_src_items)
        .set_arg(&dev)
        .set_local_work_size(local_work_size)
        .set_global_work_size(global_work_size)
        .enqueue_nd_range(&queue)?;
    let _reduce_event = ExecuteKernel::new(&reduce)
        .set_arg(&src_buffer)
        .set_arg(&dst_buffer)
        .set_global_work_size(num_groups)
        .set_wait_event(&minp_event)
        .enqueue_nd_range(&queue)?;
    queue.finish()?;
    let mut data_ptr = ptr::null_mut();
    let _dst_ev = queue.enqueue_map_buffer(
        &dst_buffer,
        CL_TRUE,
        CL_MAP_READ,
        0,
        num_groups * mem::size_of::<cl_uint>(),
        &mut data_ptr,
        &[],
    )?;
    let data: &[cl_uint] =
        unsafe { std::slice::from_raw_parts(data_ptr as *const cl_uint, num_groups) };
    let mut debug_ptr = ptr::null_mut();
    let _dbg_ev = queue.enqueue_map_buffer(
        &dbg_buffer,
        CL_TRUE,
        CL_MAP_READ,
        0,
        global_work_size * mem::size_of::<cl_uint>(),
        &mut debug_ptr,
        &[],
    )?;
    let debug: &[cl_uint] =
        unsafe { std::slice::from_raw_parts(debug_ptr as *const cl_uint, global_work_size) };

    println!(
        "{} groups, {} threads, count {}, stride {}",
        debug[0], debug[1], debug[2], debug[3]
    );
    println!("computed value: {}", data[0]);
    if data[0] == min {
        println!("result correct");
    } else {
        println!("result INcorrect");
    }

    Ok(())
}
