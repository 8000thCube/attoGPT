use std::{
	fs::File,
	io::{BufReader,BufWriter,Error as IOError,ErrorKind as IOErrorKind,Read as IORead,Write as IOWrite},
	mem::size_of
};
impl <S:Save,Z:Save>Save for (S,Z){
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{Ok((S::read(reader)?,Z::read(reader)?))}
	fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{
		let (s,z)=self;
		s.write(writer)?;
		z.write(writer)
	}
}impl <S:Save>Save for Vec<S>{
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{
		let len=usize::read(reader)?;
		let mut result=Vec::with_capacity(len);
		for _ in 0..len{result.push(S::read(reader)?)}
		Ok(result)
	}fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{
		self.len().write(writer)?;
		for x in self{x.write(writer)?}
		Ok(())
	}
}impl Save for String{
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{
		let vec=Vec::read(reader)?;
		if let Ok(string)=String::from_utf8(vec){Ok(string)}else{Err(IOError::from(IOErrorKind::InvalidData))}
	}fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{
		let bytes=self.as_bytes();
		bytes.len().write(writer)?;
		for x in bytes{x.write(writer)?}
		Ok(())
	}
}impl Save for char{
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{
		let code=u32::read(reader)?;
		if let Some(character)=Self::from_u32(code){Ok(character)}else{Err(IOError::from(IOErrorKind::InvalidData))}
	}fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{u32::from(*self).write(writer)}
}macro_rules! impl_numeric_primitive_save{
	($($t:ty),+)=>($(impl Save for $t{
		fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>{
			let mut bytes=[0;size_of::<Self>()];
			reader.read_exact(&mut bytes)?;
			Ok(<Self>::from_le_bytes(bytes))
		}fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>{writer.write_all(&self.to_le_bytes())}
	})*);
}impl_numeric_primitive_save!(f32,f64,i128,i16,i32,i64,i8,isize,u128,u16,u32,u64,u8,usize);
pub trait Save:Sized{
	fn load(filename:&str)->Result<Self,IOError>{Self::read(&mut BufReader::new(File::open(filename)?))}
	fn read<I:IORead>(reader:&mut I)->Result<Self,IOError>;
	fn save(&self,filename:&str)->Result<(),IOError>{self.write(&mut BufWriter::new(File::create(filename)?))}
	fn write<I:IOWrite>(&self,writer:&mut I)->Result<(),IOError>;
}
