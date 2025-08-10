import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  const galleryDir = path.join(process.cwd(), 'public', 'gallery');
  let images: string[] = [];
  if (fs.existsSync(galleryDir)) {
    images = fs
      .readdirSync(galleryDir)
      .filter((file) => /\.(jpe?g|png|gif|webp)$/i.test(file))
      .map((file) => `/gallery/${file}`);
  }
  return NextResponse.json(images);
}
