import fs from 'fs';
import path from 'path';

export function getGalleryImages(): string[] {
  const galleryDir = path.join(process.cwd(), 'public', 'gallery');
  if (!fs.existsSync(galleryDir)) return [];
  return fs
    .readdirSync(galleryDir)
    .filter((file) => /\.(jpe?g|png|gif|webp)$/i.test(file))
    .map((file) => `/gallery/${file}`);
}
