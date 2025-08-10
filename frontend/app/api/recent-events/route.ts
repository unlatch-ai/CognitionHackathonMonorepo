import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const eventsDir = path.join(process.cwd(), 'public', 'recent_events');
    const eventDescPath = path.join(eventsDir, 'event_desc.json');
    
    if (!fs.existsSync(eventDescPath)) {
      return NextResponse.json([]);
    }

    const eventDescData = JSON.parse(fs.readFileSync(eventDescPath, 'utf8'));
    
    // Add photos from the directory for each event
    const eventsWithPhotos = eventDescData.map((event: any) => {
      // Get all image files in the directory
      const allFiles = fs.readdirSync(eventsDir);
      const imageFiles = allFiles.filter((file) => 
        /\.(jpe?g|png|gif|webp)$/i.test(file) && file !== 'event_desc.json'
      );
      
      // Map specified photos to full paths, or use all photos if none specified
      const photos = event.photos && event.photos.length > 0 
        ? event.photos.map((photo: string) => `/recent_events/${photo}`)
        : imageFiles.slice(0, 3).map((file: string) => `/recent_events/${file}`);
        
      return {
        ...event,
        photos
      };
    });

    return NextResponse.json(eventsWithPhotos);
  } catch (error) {
    console.error('Error loading recent events:', error);
    return NextResponse.json([]);
  }
}
