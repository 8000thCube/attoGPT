mod ai;
mod save;
fn main(){
	test00();
	test01();
	test02();
}fn test00(){
	use ai::GeluMultilayerPerceptron;
	let (input,mut testnetwork,target)=([0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0, 0.5,0.5],GeluMultilayerPerceptron::new(2,10,2,1),[0.0,1.0,1.0,0.0,0.5]);
	//println!("{}",testnetwork.train_rprop(&input,0.5,1.2,1000,&target).sqrt());
	//println!("{}",testnetwork.train(&input,1000,0.5,0.01,&target).sqrt());
	println!("{}",testnetwork.train_adamw(0.9,0.999,&input,1000,0.001,&target,0.01).sqrt());
	input.chunks_exact(2).for_each(|input|println!("{}",testnetwork.process(input)[0]));
	println!("{}",testnetwork.process(&[0.75,0.75])[0]);
}fn test01(){
	use ai::SLT;
	let (input,mut testnetwork,target)=("000001010011100101110111".as_bytes(),SLT::new(3,4,2,10,1),[0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 0.0,1.0,0.0]);
	println!("{}",testnetwork.train(input,10000,0.5,0.0001,&target).sqrt());
	input.chunks_exact(3).for_each(|input|{
		testnetwork.process(input).iter().for_each(|x|print!("{} ",x));
		println!();
	});
}fn test02(){
	use ai::GPT;
	use save::Save;
	use std::fs;
	let (batchdimension,contextdimension)=(10,100);
	let (batcharea,data,mut testnetwork)=(batchdimension*contextdimension,fs::read_to_string("dataset.txt").unwrap(),GPT::load("test save spot a").unwrap_or_else(|_|GPT::load("test save spot b").unwrap_or_else(|_|GPT::new(contextdimension,4,16,4,256))));
	let (data,eval)=data.as_bytes().split_at(data.len()*9/10);
	let (databound,evalbound)=(data.len().saturating_sub(batcharea),eval.len().saturating_sub(batcharea));
	(0..10000).for_each(|n|{
		let k=ai::rsize(databound,testnetwork.seed());
		//testnetwork.train(&data[k..batcharea+k],0.9,1.0);
		//testnetwork.train_rprop(&data[k..batcharea+k],0.5,1.2);
		testnetwork.train_adamw(0.9,0.999,&data[k..batcharea+k],n,0.001,0.01);
		if n%10==0{
			let k=ai::rsize(evalbound,testnetwork.seed());
			println!("iteration: {}",n);
			println!("loss: {}",testnetwork.evaluate(&eval[k..batcharea+k]));
			println!("sample: {}",testnetwork.sample_string());
			testnetwork.save("test save spot a").unwrap();//TODO find a better way to avoid accidentally corrupting the save when terminating the program
			testnetwork.save("test save spot b").unwrap();
		}
	});
}
