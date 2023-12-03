mod ai;
mod save;
fn main(){
	test00();
	test01();
	test02();
}fn test00(){
	use ai::GeluMultilayerPerceptron;
	let (input,mut testnetwork,target)=([0.0,0.0, 0.0,1.0, 1.0,0.0, 1.0,1.0, 0.5,0.5],GeluMultilayerPerceptron::new(true,2,10,2,1),[0.0,1.0,1.0,0.0,0.5]);
	testnetwork.train(None,&input,1000,0.5,0.01,&target);
	input.chunks_exact(2).for_each(|input|println!("{}",testnetwork.process(input)[0]));
	println!("{}",testnetwork.process(&[0.75,0.75])[0]);
}fn test01(){
	use ai::SLT;
	let (input,mut testnetwork,target)=("000001010011100101110111".as_bytes(),SLT::new(3,2,10,1,256),[0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,1.0, 0.0,0.0,1.0, 0.0,1.0,1.0, 0.0,1.0,1.0, 0.0,1.0,0.0, 0.0,1.0,0.0]);
	testnetwork.train(None,input.iter().map(|&x|x as usize),10000,0.5,0.001,&target);
	input.chunks_exact(3).for_each(|input|{
		testnetwork.process(input.iter().map(|&x|x as usize)).iter().for_each(|x|print!("{} ",x));
		println!();
	});
}fn test02(){//ln(256)=5.545177 //lg(e)=1.442695
	use ai::{GPT,NeuralNetwork,rseed,rsize};//h: 2.19
	use save::Save;
	use std::fs;
	let (batchdimension,contextdimension,lossexamplecount,mut n,mut seed)=(10,100,10,0,rseed());
	let (data,exampledimension,mut derivatives,mut testnetwork)=(fs::read_to_string("dataset.txt").unwrap(),contextdimension+1,NeuralNetwork::default(),GPT::load("test save spot a2").unwrap_or_else(|_|GPT::new(true,4,4,16,256,256)));
	let (data,eval)=data.as_bytes().split_at(data.len()*9/10);
	let (databound,evalbound)=(data.len().checked_sub(exampledimension).unwrap(),eval.len().checked_sub(exampledimension).unwrap());
	(0..10000).for_each(|_|{
		testnetwork.train(batchdimension,exampledimension,Some(&mut derivatives),|network,networkderivatives|{
			network.adamw(0.9,0.999,networkderivatives,0.00001,n,0.01);
			//network.rprop(networkderivatives,0.8,1.25);
			n+=1;
		},1.0,(0..batchdimension*10).flat_map(move|_|{
			let k=rsize(databound,&mut seed);
			data[k..exampledimension+k].iter().map(|&x|x as usize)
		}));
		println!("iteration: {}",n);
		println!("loss: {}",(0..lossexamplecount).map(|_|{
			let k=rsize(evalbound,&mut seed);
			testnetwork.evaluate(1.0,eval[k..exampledimension+k].iter().map(|&x|x as usize))
		}).sum::<f32>()/lossexamplecount as f32);
		println!("sample:\n{}",testnetwork.sample_string(contextdimension,contextdimension*2,1.0));
		testnetwork.save("test save spot a2").unwrap();
	});
}
