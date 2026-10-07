# NHL ’94: Player Strategy and AI Design Guide

Consolidated from four supplied NHL ’94 podcast transcripts, including two substantially duplicate transcriptions. Updated October 6, 2026.

**The central lesson is to create favorable situations through skating, positioning, player selection and timing.** Finishing a chance is only one part of a possession. A useful AI must also recover the puck, move it safely, create an opening and select a finish that suits the opponent.

This guide primarily concerns the original **Sega Genesis** game. The first two files appear to transcribe the same Angryjay93 interview, so repeated claims in them are not independent confirmation. The Adams Division episode adds team-specific examples and lineup choices. The fourth file is part two of Len’s discussion with guest EA; it adds EA-special practice, one-timer setups, defensive recovery and experiences with different ROMs. Player observations below are attributed to the transcripts; AI applications are proposed experiments, not results demonstrated by the speakers.

Source labels used throughout:

- **[S1]** `Angryjay93 - Line Combos & Crushing.txt` — general strategy, practice, attributes, checking, scoring and goalie play.
- **[S2]** `Angryjay93, Teaching The Art Of Line Combinations.txt` — substantially the same interview, with different transcription errors.
- **[S3]** `How to Build the Best NHL '94 Lines ft. Angryjay93 (Adams Division).txt` — Boston, Buffalo, Hartford, Montreal, Ottawa and Quebec.
- **[S4]** `Pasted text(2).txt` — part two of Len’s conversation with EA: rebound practice and risk, passing combinations, recovery through player switching, constrained lineups and transfer between game variants. The file does not supply a full episode title or speaker labels.
- **[W1]** [Insider Weight Knowledge! (Weight Bug)](https://forum.nhl94.com/index.php?/topic/16081-insider-weight-knowledge-weight-bug/) — community testing documenting the distinction between user-controlled and built-in CPU checking.
- **[W2]** [Closing the gap between GENS A and B players](https://forum.nhl94.com/index.php?/topic/17911-closing-the-gap-between-gens-a-and-b-players/) — additional player advice on positioning, stopping and starting, scoring practice and manual-goalie mixups.

Player names are normalized where the intended identity is clear. The transcripts contain speech-recognition errors and references to an on-screen roster that is not included. Lineups below are starting points described in the discussion, not universal rankings or verified optimal solutions. Tournament tiers and matchup preferences refer to the episode’s context. Custom-league rosters, eligibility restrictions and altered ratings discussed in [S4] do not replace the original-roster lineup advice in [S3].

**For playing better, the most transferable tactics are these:**

| Tactic | Practical application | Implication for an AI |
|---|---|---|
| Influence computer-controlled movement | Experiment with stopping near the blue line, changing your route, and carrying on the opposite wing. Watch where teammates and defenders move in response. | Predict movement conditional on the action you take. Judge a passing lane at the expected arrival time, not only at release. |
| Draw pressure before passing | Make the puck carrier a threat so a defender commits and another attacker becomes available. | Learn when carrying creates more value than an immediate pass. |
| Use mobile defensemen selectively | Angryjay describes Coffey as an extra forward: he carries while the three forwards seek openings. In [S4], EA describes difficulty with that approach and a preference for keeping defensemen back. | Compare the extra offensive opportunity with recovery time and counterattack exposure. Evaluate the tactic in the context of the lineup and controller’s ability. |
| Preserve defensive position | Use poke checks when a body check is unreliable or a miss would open a dangerous route. | Optimize puck recovery and chance prevention. A hit is useful only through its consequences. |
| Change how you enter the zone | Against pressure at the blue line, an occasional dump after crossing the red line can change the defensive response and let forwards pursue. | Choose carrying, passing or dumping according to pressure and recovery prospects. |
| Plan for rebounds and possible failure | The “EA special” creates a rebound and goalie displacement that can open a follow-up finish. Off-wing forehand setups recur in the interviews. Check whether your defense can recover if you lose the rebound. | Train approach, first shot, pursuit, finish and the response to losing possession. Judge the whole sequence, including the opponent’s counterattack. |
| Use passing combinations to move the goalie | Skate to create a teammate opening, then consider a return pass before the one-timer if a manual goalie commits to the first receiver. | Choose between an immediate finish and another pass using receiver movement, goalie position and interception risk. |
| Recover through deliberate player switching | EA describes selecting successive skaters and giving them speed bursts to help restore defensive position. | Track the selected player and movement direction after every switch; measure whether coverage actually improves. |
| Maintain alternative finishes | A manual goalie may dive early, stay deep or anticipate your usual shot. A single automatic response becomes predictable. | Retain several finishes from similar approaches and adapt to observed goalie behavior. |
| Match the player to the task | Skating, weight, handedness, puck skills and shooting determine which opportunities a player can exploit. | Condition behavior on the current player’s capabilities and the opposing matchup. |

The original tactics come from [S1–S2]; [S4] adds the rebound-risk qualification, passing combinations and recovery example. The advice on drawing defenders, stopping and starting, and varying finishes is also supported by the player discussion in [W2].

**The EA special has several reported setups, and the new interview makes its tactical cost clearer.** In [S1–S2], Angryjay describes an off-wing approach and often releasing the shot without a held direction. In [S4], Len describes moving inward from the faceoff dot toward the slot and aiming toward the far upper corner. EA emphasizes the blade orientation and sending the puck toward a place the goalie must move to cover. These are player-specific accounts, not one verified universal input sequence.

The common objective is to create a rebound and an opening, then reach the puck before the defense. Len says his choice of setup depends mainly on his position, handedness and nearby defenders; that observation does not establish that goalie identity or positioning is irrelevant. A right-handed shooter on the left, or left-handed shooter on the right, is his preferred arrangement. He finds the corresponding backhand attempts less consistent. [S4]

More importantly, Len reports that a failed attempt during a rush can leave his defensemen still skating forward and give the opponent a breakaway. He therefore prefers trying it after the attacking formation has settled. The transferable decision is to inspect **defender position and velocity, likely rebound recovery and the opponent’s outlet**, rather than treating every usable shooting angle as a good opportunity. A defender near the blue line but moving toward the opposing net may provide less useful cover than its position alone suggests. [S4; the state features are an AI-design inference.]

**One-timers can be created through movement and an extra pass.** EA praises Len’s ability to skate through positions so a teammate crosses into an available lane, sometimes after a spin or turn. Len also describes learning back-and-forth passing from New Jersey Killer: a manual goalie prepares for one receiver, then the puck returns for a finish elsewhere. For practice, work on the teammate movement and pass timing as well as the final shot. For an AI, predict receiver and goalie movement and retain the option to shoot immediately when another pass adds needless risk. Len acknowledges that pursuing an attractive combination can sometimes replace an easier goal. [S4]

**Defensive recovery is a sequence of control decisions.** EA describes repeated C-then-B inputs to boost and switch among skaters while getting them back. This use of switching is distinct in purpose from the contact-based CB checking technique discussed elsewhere. It is sequential control of different skaters, not simultaneous direct control of the whole team. He also reports failures when a burst catches a player turning or sends someone out of position. Treat it as a behavior to test with direction, momentum, possession and selected-player feedback, not a blind repeating macro. [S4]

**The Adams Division episode sharpens the earlier summary: lineup quality depends on how players’ jobs fit together.** A team needs some combination of puck recovery, safe exits, zone entry, chance creation, finishing and defensive coverage. More highly rated shooters do not automatically solve a shortage of puck recovery or mobility. [S3]

Five examples make that lesson concrete:

1. **Buffalo’s transition has an extra step.** Angryjay describes Mogilny as exceptionally dangerous with possession, but less convenient for winning it through checks. LaFontaine or Svoboda may first need to recover the puck and then find him. Evaluate the recovery-to-outlet sequence as a whole; the star’s scoring ability does not guarantee an easy counterattack.
2. **Speed differences can disrupt combinations.** Mogilny can get ahead of the slower supporting forwards, making some one-timer setups awkward. A fast player, a good passer and a good shooter do not guarantee that the receiver will be in the right place at the right time.
3. **Quebec benefits from a different kind of player.** Many forwards are primarily shooters. Kovalenko adds puck recovery and defensive disruption; Tatarinov adds a different defensive and puck-moving option. Their contribution can exceed that of another player with better-looking offensive ratings.
4. **Montreal’s defense helps produce its offense.** Roy and the defensive unit can encourage an opponent to commit more attackers. The agile forwards can then counter into the space left behind. Leading the game changes the opportunities available.
5. **Control mode and assignments matter.** A heavy finisher such as Neely can contribute through computer-controlled checking, positioning and finishing without being your main puck carrier. Hartford’s Zalapski can be placed on the side facing the opponent’s stronger winger. The player you control and the players you leave to the game’s AI are part of the tactic.

**The following Adams Division lineups are useful places to start experimenting.** Forward positions are shown explicitly when the speaker provides them. Defensive pairings are unordered unless sides are specified. [S3]

| Team | Forward starting point | Defense and goalie | Intended style and main adjustment |
|---|---|---|---|
| **Boston Bruins** | LW Adam Oates; C Joe Juneau; RW Ted Donato. Cam Neely is the main shooting alternative. | Ray Bourque and Don Sweeney; Andy Moog. Glen Wesley and Gord Murphy provide alternatives on defense. | Build a lead and defend it. Oates creates, Juneau supplies mobility and two-way play, and Donato disrupts. Add Neely when finishing matters more, but avoid relying on him to carry through pressure. |
| **Buffalo Sabres** | Common setup: LW Pat LaFontaine; C Alexander Mogilny; RW Dale Hawerchuk. Yuri Khmylev offers a shooting alternative. | Petr Svoboda and Richard Smehlik; Grant Fuhr. | Recover possession, find Mogilny and attack quickly. Support the weaker second defenseman. Injuries or penalties expose limited depth, so the plan needs to change when a key player is unavailable. |
| **Hartford Whalers** | Pat Verbeek and Geoff Sanderson can exchange forward roles; Terry Yake generally stays on a wing. Yake at RW is one discussed option. | LD Eric Weinrich or Adam Burt; RD Zarley Zalapski; Sean Burke. | Use a balanced group with few glaring weaknesses in its usual matchups. Put Zalapski opposite the threatening left winger. Consider Michael Nylander as a lighter matchup option. |
| **Montreal Canadiens** | Common setup: LW Denis Savard; C Stephan Lebeau or Kirk Muller; RW Vincent Damphousse. | A starting pair of LD J.J. Daigneault and RD Eric Desjardins; Patrick Roy. Patrice Brisebois and Mathieu Schneider offer alternatives. | Defend, recover and counter together. Lebeau supports a lighter lineup; Muller adds shot power and heavier checking options. Manage Daigneault’s weak passing when breaking out. |
| **Ottawa Senators** | LW Sylvain Turgeon; C Jamie Baker; RW Doug Smail. Bob Kudelski at C provides a shooting alternative. | LD Norm Maciver; RD Brad Shaw. Brad Marsh is a situational heavier checking option. Peter Sidorkiewicz versus Daniel Berthiaume is a goalie tradeoff. | Give each limited player a manageable job. Preserve Shaw’s position and use Maciver for movement and exits. Berthiaume offers manual mobility, with the crease-contact vulnerability described in the interview. |
| **Quebec Nordiques** | LW Mats Sundin; C Joe Sakic; RW Andrei Kovalenko. Valeri Kamensky and Owen Nolan are additional offensive options. | Mikhail Tatarinov and Steve Duchesne are Angryjay’s starting pair; Ron Hextall. | Use Kovalenko to recover and disrupt, then move the puck quickly to the shooters. The difficult parts are defending and reaching the attacking zone consistently. |

Several details in the team discussion apply beyond those six teams:

- **Match the defensive tool to the matchup.** Similar-weight opponents can make a poke check and containment preferable to a committed body check. A specialist only earns a lineup spot if the opponents allow that specialty to matter.
- **Choose sides deliberately.** Off-wing shooters can suit particular shooting and rebound angles. Natural-wing playmakers can suit passing. A defensive matchup may justify overriding a preferred handedness arrangement.
- **Plan substitutes in advance.** The default replacement may be much worse for the job than another available player. Preserve the lost role where possible, rather than simply choosing the next overall rating.
- **Judge goalies in the way you use them.** Manual movement, stopping, starting and positioning matter alongside automatic saves. Moog illustrates why the displayed overall rating is an incomplete guide.
- **Account for score and clock.** Boston and Hartford are described as more comfortable protecting leads than chasing games. Ottawa has a more offensive combination to consider when a late goal is necessary.
- **Account for the opponent’s strengths.** A lineup and style that work against one team may fail against a faster or harder-shooting opponent. Home/away effects and hot/cold variation are also discussed, but their precise implementation requires separate verification.

These are expert judgments from [S1–S3], not a claim that one lineup will always win.

**For an NHL ’94 AI, I would translate the combined advice into a small set of tactical skills and explicit evaluation problems.** The following is a proposed development approach.

| Skill | What it must recognize and execute | Useful measurements |
|---|---|---|
| Contain and recover | Approach safely; choose poke, body check, CB check, interception or continued containment. | Possession regained, dangerous chances conceded, penalties and position after a miss. |
| Recover and release | Select an outlet after gaining the puck, including finding a stronger puck carrier. | Retained possession through the outlet, time to a safe exit, turnover location. |
| Enter the zone | Carry, pass, delay or dump according to defensive pressure and teammate support. | Controlled entries, recoveries after dumps, immediate turnovers and subsequent chances. |
| Draw and distribute | Carry, turn or stop until a useful receiver becomes available; choose an immediate finish or a return-pass combination. | Dangerous completed passes, goalie displacement and eventual goals, balanced against turnovers and missed immediate chances. |
| Create and finish a rebound | Select a suitable initial shot, check defensive cover, pursue the resulting puck and recover if possession is lost. | Whole-sequence conversion, lost rebounds and subsequent breakaways or goals conceded. |
| Finish against a goalie | Choose among available shots, dekes, one-timers and delays. | Conversion against several goalie behaviors and opponents excluded from training. |
| Switch control and cover | Decide which skater or goalie needs direct control; move successive defenders back after an attack breaks down. | Time to restore coverage, threats prevented, reaction delay, wasted bursts and openings created by switching away. |
| Take and use manual goalie control | Recognize when intervention is useful and execute positioning with the current goalie. | Saves and goals conceded after takeover, compared with leaving the built-in goalie in control in matched situations. |

A tactical selector could choose the current objective while a lower-level controller executes skating and button timing. Objectives should have flexible durations: a “draw and pass” attempt can end when the lane opens, pressure becomes unsafe, or possession changes. The controller must still handle short timing windows. None of the transcripts establishes an optimal decision frequency or neural-network architecture.

**Separate skills do not require a separate network for every move.** A shared policy conditioned on the skill, player and game situation is a reasonable first experiment. Separate networks are another option if skills interfere during training. The evidence here motivates the behavioral decomposition, not a particular model count.

An observation design should expose or estimate the information needed for those decisions: puck and player positions and velocities, possession, controlled-player identity, handedness and attributes, available teammates, score, remaining time and manpower. The failed-rebound example makes defender velocity especially important; the return-pass example makes goalie motion important. Recent observations can help infer movement and opponent tendencies. If the agent uses pixels, off-screen positions and hidden state must be estimated; RAM-based training or evaluation should explicitly define which information is allowed.

For checking, keep one distinction explicit: **a learned agent sending controller inputs occupies the user-controlled slot in the game.** It does not automatically receive the behavior of NHL ’94’s built-in CPU players. Community testing describes different weight-related checking behavior for those two modes and the C-then-B technique that changes control before contact. Track the relevant control transition when implementing or testing this mechanic. [W1]

**Training should combine repeatable practice with opportunities that emerge during play.** Angryjay values save-state repetition for execution, but also emphasizes practicing moves in naturally developing situations. [S1–S2]

Len supplies a concrete practice method in [S4]: he played CPU playoff series with Ottawa and then Anaheim, attempting the EA special on every shooting opportunity regardless of the score. He found the slower players and shots easier to read. This describes his experience with those teams, not emulator slowdown or a guarantee that weak teams are easier overall. It is a useful skill-acquisition drill; normal competitive play should restore the choice to pass, delay or use a different finish.

1. Build a collection of starting situations for each skill, varying positions, velocities, players, defenders and attack direction. Avoid learning only one exact save state.
2. Begin a difficult execution skill in readable situations, such as the slower-player rebound practice described in [S4], then test faster players and tighter pressure. Also train the decision to use it: a rebound routine must recognize useful cover; a body check must recognize when committing is worthwhile.
3. Chain the parts. Buffalo is a useful case for recovery, outlet to Mogilny and entry. Quebec tests whether puck recovery and passing can activate strong finishers.
4. Evaluate in continuous games against varied defensive and goalie behavior, including opponents excluded from training. Success against the built-in goalie alone does not establish competitive strength.
5. Keep match outcomes as the final measure. Skill metrics diagnose weaknesses; rewards for merely passing, hitting or shooting can encourage activity that does not help win.
6. Expose skills that strong teammates can hide. Len says relying on a strong automatic goalie reduced his manual-goalie practice. Use targeted takeover situations and varied goalies, then verify that intervention improves outcomes. Treat weak-roster or difficult away-game practice as a stress test, not the only training distribution. [S4]

The lineup discussions in [S3–S4] also suggest testing robustness. Compare performance with a normal lineup, a missing star, a substitute defender and a different opposing weight/speed profile. For an agent intended to play multiple teams, it should learn when to change its behavior rather than execute the same plan with every roster.

**Keep the target ROM central to training and evaluation.** The speakers in [S4] describe altered goalie movement, control timing and highly boosted players in custom leagues as changing their habits. One speaker reports that SNES practice broadened his Genesis deking repertoire, while the other finds switching versions difficult. These anecdotes motivate measuring transfer in both directions; they do not establish that another version or league reliably improves or harms skill. Record the ROM and gameplay settings for each experiment, keep evaluation on the intended target, and test altered-game training against a target-only baseline. Claims about fast passes bouncing off skates remain observations rather than an established engine explanation.

**A focused first experiment is deliberate chance creation.** Start from varied attacking situations with an available teammate and a defender obstructing the useful pass. Allow the agent to carry, turn, stop or pass. Compare it with a baseline that immediately passes to the teammate who currently looks most open. Measure dangerous receptions, turnovers and goals over the whole possession. This tests whether it learns to create an opening instead of only reacting to an existing one.

**A second experiment from [S4] is deciding when to attempt an EA special.** Build situations with a similar shooting angle but different defensive cover: a settled attack, defenders moving forward, an opponent ready for an outlet, and pressure near the rebound area. From each saved state, compare attempting the rebound play with passing or retaining possession. Measure goals for, lost rebounds, breakaways conceded and goals against through the following transition. Compare a policy that always attempts the move from that angle with one that can decline it. This tests tactical selection separately from the ability to execute the shot.

**Some claims should remain hypotheses until tested in the exact ROM.** The interviews are valuable strategy evidence, but they are not an engine specification.

| Claim or uncertainty | How to handle it |
|---|---|
| Original Genesis weight behavior versus SNES or modified ROMs | Identify the version and test representative user-controlled, CPU-controlled and CB checks. Do not transfer rules automatically between versions. |
| Repeat-check immunity after a resisted check | Test the affected player pair, contact history, reset conditions and different checking actions. Do not assume the transcript gives a complete state-machine description. |
| Pulling the goalie increases shot accuracy | Compare otherwise matched shot setups with and without the goalie, across varied states. Do not treat the interview’s strong wording as proof of guaranteed goals. |
| Awareness ratings determine precise positioning behavior | Measure teammate movement and availability under controlled changes. The general interview acknowledges uncertainty; EA’s suggestion in [S4] that low defensive awareness delays a positional switch is explicitly a theory. Do not adopt “low awareness means better defensive positioning” as a rule. |
| One-timers reduce the goalie’s save area | [S4] refers secondhand to an analysis by Chaos. The analysis itself is not supplied here. Check the original work or code and reproduce the conditions before treating the explanation as established. |
| Particular goalie-contact angles, lofted shots or button combinations reliably force goals | EA describes exploratory penalty-shot and crease-contact ideas in [S4]. Test geometry, timing and failure cases before promoting them to dependable scoring skills; no reliable recipe is established by this transcript. |
| Screens or apparent momentum change outcomes | Separate observed outcomes from explanations. The speakers do not establish a definitive Genesis screening or momentum mechanism. |
| Home/away, hot/cold and handedness affect performance | Evaluate both rink directions, both home/away assignments and varied initial states. Avoid converting anecdotes or displayed modifiers into unsupported numerical rules. |

For human practice, the most useful starting sequence is defensive positioning, puck control with changes of pace, a dependable scoring method with an alternative finish, and then team-specific lineups. For AI development, use the transcripts to define skills, situations, matchup features and experiments. They do not supply the synchronized observations and controller inputs required for direct action imitation.
