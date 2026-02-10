import { Router, Request, Response } from 'express';
import { createMatch, getMatch, listMatches } from './matchStore';

const router = Router();

// Create a new match
router.post('/matches', (req: Request, res: Response) => {
  const { title, teamA, teamB, sport, venue } = req.body;
  if (!title || !teamA || !teamB) {
    res.status(400).json({ error: 'title, teamA, and teamB are required' });
    return;
  }
  const match = createMatch({ title, teamA, teamB, sport, venue });
  res.status(201).json(match);
});

// List all matches
router.get('/matches', (_req: Request, res: Response) => {
  res.json(listMatches());
});

// Get match by code
router.get('/matches/:code', (req: Request<{ code: string }>, res: Response) => {
  const match = getMatch(req.params.code);
  if (!match) {
    res.status(404).json({ error: 'match not found' });
    return;
  }
  res.json(match);
});

export default router;
