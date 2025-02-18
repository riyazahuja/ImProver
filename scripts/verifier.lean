import TrainingData.Frontend
import Cli
import scripts.state_comments

open Lean Core Elab IO Meta Term Tactic

set_option autoImplicit true



def insert_state_comments (step:CompilationStep) : IO String := do
  let mut trees := step.trees
  trees := trees.flatMap InfoTree.retainTacticInfo
  trees := trees.flatMap InfoTree.retainOriginal
  trees := trees.flatMap InfoTree.retainSubstantive

  let L₁ ← (trees.flatMap InfoTree.tactics).mapM TacticInvocation.rangeAndStates
  let L₂ := dropEnclosed L₁ |>.filter fun ⟨⟨⟨l₁, _⟩, ⟨l₂, _⟩⟩, _, _⟩  => l₁ = l₂
  let L₃ := (L₂.map fun ⟨r, sb, sa⟩ => (r, formatState sb, formatState sa))
  let mut src := ({str:=step.src.str, stopPos := step.src.stopPos, startPos := 0} : Substring).toString.splitOn "\n"
  let mut inserted : Std.HashSet Nat := Std.HashSet.ofList [10000000]
  for item in L₃.reverse do
    let ⟨⟨⟨l, c⟩, _⟩, sb, sa⟩ := item
    if sa.contains "🎉 no goals" then
      src := src.insertIdx l $ stateComment sa c
    if inserted.contains (l-1) then
      src := src.set (l-1) $ stateComment sb c
    else
      src := src.insertIdx (l-1) $ stateComment sb c
      inserted := inserted.insert (l-1)

  let out := ("\n".intercalate src)
  return out

/-- Our verifier needs two parts: First evaluate a whole module with the proof as sorry option on.
this should return a bunch of compilationSteps. Then on each decl in this module, we run the "improvement loop"

This "loop" for now will literally just print out the
theorem with the proof states interleaved.

Then we will elaborate this string on the environment before the command which created that declaration.
then we will print out the theorem with the proof states interleaved.
--/

def runAtDecls (mod : Name) : IO Unit := do
  let proofAsSorry := ({} : KVMap).insert `debug.proofAsSorry (.ofBool true)
  let steps := Lean.Elab.IO.processInput' (← moduleSource mod) none proofAsSorry (← findLean mod).toString

  let targets := steps.bind fun c => (MLList.ofList c.diff).map fun i => (c, i)

  for (cmd, ci) in targets do
    for m in cmd.msgs do IO.eprintln (bombEmoji ++ (← m.data.toString))
    unless cmd.msgs.isEmpty do
      throw <| IO.userError s!"Unexpected messages in: {mod} during elaboration of {cmd.stx}"

    let contents := cmd.src.toString
    IO.println s!"COMPILATION STEP CONTENTS:\n {contents}"
    let prev_state := cmd.before
    --now, let's add the comment to the theorem, and run it after the prev env
    let elaborated_steps := Lean.Elab.IO.processInput' contents (some prev_state) {}

    let head? ← elaborated_steps.uncons
    match head? with
    | none =>
      IO.println s!"No elaborated steps"
    | some (head, _) =>
      -- Should probably check that ci is actually in the diff? But the below code is a bit finnicky with namespaces.
      -- works fine without it anyways

      -- if not ((head.diff.map (fun info=>info.name)).contains ci.name) then
      --   IO.eprintln s!"Expected {ci.name} to be in the elaborated steps, but it was not"
      -- else
      --   IO.println s!"Found {ci.name} in the elaborated steps"
        IO.println s!"AFTER ELAB CONTENTS:\n {← insert_state_comments head}"

/-
COMPILATION STEP CONTENTS:
 /-- The relation that specifies valid moves in our hydra game. `CutExpand r s' s`
  means that `s'` is obtained by removing one head `a ∈ s` and adding back an arbitrary
  multiset `t` of heads such that all `a' ∈ t` satisfy `r a' a`.

  This is most directly translated into `s' = s.erase a + t`, but `Multiset.erase` requires
  `DecidableEq α`, so we use the equivalent condition `s' + {a} = s + t` instead, which
  is also easier to verify for explicit multisets `s'`, `s` and `t`.

  We also don't include the condition `a ∈ s` because `s' + {a} = s + t` already
  guarantees `a ∈ s + t`, and if `r` is irreflexive then `a ∉ t`, which is the
  case when `r` is well-founded, the case we are primarily interested in.

  The lemma `Relation.cutExpand_iff` below converts between this convenient definition
  and the direct translation when `r` is irreflexive. -/
def CutExpand (r : α → α → Prop) (s' s : Multiset α) : Prop :=
  ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ s' + {a} = s + t


AFTER ELAB CONTENTS:
 /-- The relation that specifies valid moves in our hydra game. `CutExpand r s' s`
  means that `s'` is obtained by removing one head `a ∈ s` and adding back an arbitrary
  multiset `t` of heads such that all `a' ∈ t` satisfy `r a' a`.

  This is most directly translated into `s' = s.erase a + t`, but `Multiset.erase` requires
  `DecidableEq α`, so we use the equivalent condition `s' + {a} = s + t` instead, which
  is also easier to verify for explicit multisets `s'`, `s` and `t`.

  We also don't include the condition `a ∈ s` because `s' + {a} = s + t` already
  guarantees `a ∈ s + t`, and if `r` is irreflexive then `a ∉ t`, which is the
  case when `r` is well-founded, the case we are primarily interested in.

  The lemma `Relation.cutExpand_iff` below converts between this convenient definition
  and the direct translation when `r` is irreflexive. -/
def CutExpand (r : α → α → Prop) (s' s : Multiset α) : Prop :=
  ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ s' + {a} = s + t


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_le_invImage_lex [DecidableEq α] [IsIrrefl α r] :
    CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ (· ≠ ·)) (· < ·)) toFinsupp := by
  rintro s t ⟨u, a, hr, he⟩
  replace hr := fun a' ↦ mt (hr a')
  classical
  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)]
      using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)),
      add_zero] at he
    exact he ▸ Nat.lt_succ_self _


AFTER ELAB CONTENTS:
 theorem cutExpand_le_invImage_lex [DecidableEq α] [IsIrrefl α r] :
    CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ (· ≠ ·)) (· < ·)) toFinsupp := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    ι✝ : Type u_3
    α✝ : Type u_4
    inst✝² : Zero α✝
    toFinsupp : ι✝ → Finsupp α α✝
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    ⊢ LE.le (sorryAx (ι✝ → ι✝ → Prop) Bool.true) (InvImage (Finsupp.Lex (Min.min ( …
  -/
  rintro s t ⟨u, a, hr, he⟩
  /-
    🎉 no goals
  -/
  replace hr := fun a' ↦ mt (hr a')
  classical
  refine ⟨a, fun b h ↦ ?_, ?_⟩ <;> simp_rw [toFinsupp_apply]
  · apply_fun count b at he
    simpa only [count_add, count_singleton, if_neg h.2, add_zero, count_eq_zero.2 (hr b h.1)]
      using he
  · apply_fun count a at he
    simp only [count_add, count_singleton_self, count_eq_zero.2 (hr _ (irrefl_of r a)),
      add_zero] at he
    exact he ▸ Nat.lt_succ_self _


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_singleton {s x} (h : ∀ x' ∈ s, r x' x) : CutExpand r s {x} :=
  ⟨s, x, h, add_comm s _⟩


AFTER ELAB CONTENTS:
 theorem cutExpand_singleton {s x} (h : ∀ x' ∈ s, r x' x) : CutExpand r s {x} :=
  ⟨s, x, h, add_comm s _⟩


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_singleton_singleton {x' x} (h : r x' x) : CutExpand r {x'} {x} :=
  cutExpand_singleton fun a h ↦ by rwa [mem_singleton.1 h]


AFTER ELAB CONTENTS:
 theorem cutExpand_singleton_singleton {x' x} (h : r x' x) : CutExpand r {x'} {x} :=
  cutExpand_singleton fun a h ↦ by rwa [mem_singleton.1 h]


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_add_left {t u} (s) : CutExpand r (s + t) (s + u) ↔ CutExpand r t u :=
  exists₂_congr fun _ _ ↦ and_congr Iff.rfl <| by rw [add_assoc, add_assoc, add_left_cancel_iff]


AFTER ELAB CONTENTS:
 theorem cutExpand_add_left {t u} (s) : CutExpand r (s + t) (s + u) ↔ CutExpand r t u :=
                                                  /-
                                                    x✝² : Sort u_1
                                                    CutExpand : x✝²
                                                    t : ?m.134
                                                    u : ?m.135
                                                    s : ?m.136
                                                    x✝¹ : ?m.145
                                                    x✝ : ?m.146 x✝¹
                                                    ⊢ Iff (?m.164 x✝¹ x✝) (?m.168 x✝¹ x✝)
                                                  -/
  exists₂_congr fun _ _ ↦ and_congr Iff.rfl <| by rw [add_assoc, add_assoc, add_left_cancel_iff]


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_add_right {s' s} (t) : CutExpand r (s' + t) (s + t) ↔ CutExpand r s' s := by
  convert cutExpand_add_left t using 2 <;> apply add_comm


AFTER ELAB CONTENTS:
 lemma cutExpand_add_right {s' s} (t) : CutExpand r (s' + t) (s + t) ↔ CutExpand r s' s := by
  /-
    x✝ : Sort u_1
    CutExpand : x✝
    s' : ?m.134
    s : ?m.135
    t : ?m.136
    ⊢ Iff (sorryAx Prop Bool.true) (sorryAx Prop Bool.true)
  -/
  convert cutExpand_add_left t using 2 <;> apply add_comm
  /-
    🎉 no goals
  -/


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  simp_rw [CutExpand, add_singleton_eq_iff]
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
  · rintro ⟨ht, ha, rfl⟩
    obtain h | h := mem_add.1 ha
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
  · rintro ⟨ht, h, rfl⟩
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩


AFTER ELAB CONTENTS:
 theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    s' s : Multiset α
    ⊢ Iff (sorryAx Prop Bool.true) (Exists fun t => Exists fun a => And (∀ (a' : α …
  -/
  simp_rw [CutExpand, add_singleton_eq_iff]
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
  · rintro ⟨ht, ha, rfl⟩
    obtain h | h := mem_add.1 ha
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
  · rintro ⟨ht, h, rfl⟩
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  simp_rw [CutExpand, add_singleton_eq_iff]
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
  · rintro ⟨ht, ha, rfl⟩
    obtain h | h := mem_add.1 ha
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
  · rintro ⟨ht, h, rfl⟩
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩


AFTER ELAB CONTENTS:
 theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    s' s : Multiset α
    ⊢ Iff (sorryAx Prop Bool.true) (Exists fun t => Exists fun a => And (∀ (a' : α …
  -/
  simp_rw [CutExpand, add_singleton_eq_iff]
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
  · rintro ⟨ht, ha, rfl⟩
    obtain h | h := mem_add.1 ha
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
  · rintro ⟨ht, h, rfl⟩
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩


-- Hi
COMPILATION STEP CONTENTS:
 theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  simp_rw [CutExpand, add_singleton_eq_iff]
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
  · rintro ⟨ht, ha, rfl⟩
    obtain h | h := mem_add.1 ha
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
  · rintro ⟨ht, h, rfl⟩
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩


AFTER ELAB CONTENTS:
 theorem cutExpand_iff [DecidableEq α] [IsIrrefl α r] {s' s : Multiset α} :
    CutExpand r s' s ↔
      ∃ (t : Multiset α) (a : α), (∀ a' ∈ t, r a' a) ∧ a ∈ s ∧ s' = s.erase a + t := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    s' s : Multiset α
    ⊢ Iff (sorryAx Prop Bool.true) (Exists fun t => Exists fun a => And (∀ (a' : α …
  -/
  simp_rw [CutExpand, add_singleton_eq_iff]
  refine exists₂_congr fun t a ↦ ⟨?_, ?_⟩
  · rintro ⟨ht, ha, rfl⟩
    obtain h | h := mem_add.1 ha
    exacts [⟨ht, h, erase_add_left_pos t h⟩, (@irrefl α r _ a (ht a h)).elim]
  · rintro ⟨ht, h, rfl⟩
    exact ⟨ht, mem_add.2 (Or.inl h), (erase_add_left_pos t h).symm⟩


-- Hi
COMPILATION STEP CONTENTS:
 theorem not_cutExpand_zero [IsIrrefl α r] (s) : ¬CutExpand r s 0 := by
  classical
  rw [cutExpand_iff]
  rintro ⟨_, _, _, ⟨⟩, _⟩


AFTER ELAB CONTENTS:
 theorem not_cutExpand_zero [IsIrrefl α r] (s) : ¬CutExpand r s 0 := by
  classical
  rw [cutExpand_iff]
  rintro ⟨_, _, _, ⟨⟩, _⟩


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_zero {x} : CutExpand r 0 {x} := ⟨0, x, nofun, add_comm 0 _⟩


AFTER ELAB CONTENTS:
 lemma cutExpand_zero {x} : CutExpand r 0 {x} := ⟨0, x, nofun, add_comm 0 _⟩


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_zero {x} : CutExpand r 0 {x} := ⟨0, x, nofun, add_comm 0 _⟩


AFTER ELAB CONTENTS:
 lemma cutExpand_zero {x} : CutExpand r 0 {x} := ⟨0, x, nofun, add_comm 0 _⟩


-- Hi
COMPILATION STEP CONTENTS:
 /-- For any relation `r` on `α`, multiset addition `Multiset α × Multiset α → Multiset α` is a
  fibration between the game sum of `CutExpand r` with itself and `CutExpand r` itself. -/
theorem cutExpand_fibration (r : α → α → Prop) :
    Fibration (GameAdd (CutExpand r) (CutExpand r)) (CutExpand r) fun s ↦ s.1 + s.2 := by
  rintro ⟨s₁, s₂⟩ s ⟨t, a, hr, he⟩; dsimp at he ⊢
  classical
  obtain ⟨ha, rfl⟩ := add_singleton_eq_iff.1 he
  rw [add_assoc, mem_add] at ha
  obtain h | h := ha
  · refine ⟨(s₁.erase a + t, s₂), GameAdd.fst ⟨t, a, hr, ?_⟩, ?_⟩
    · rw [add_comm, ← add_assoc, singleton_add, cons_erase h]
    · rw [add_assoc s₁, erase_add_left_pos _ h, add_right_comm, add_assoc]
  · refine ⟨(s₁, (s₂ + t).erase a), GameAdd.snd ⟨t, a, hr, ?_⟩, ?_⟩
    · rw [add_comm, singleton_add, cons_erase h]
    · rw [add_assoc, erase_add_right_pos _ h]


AFTER ELAB CONTENTS:
 /-- For any relation `r` on `α`, multiset addition `Multiset α × Multiset α → Multiset α` is a
  fibration between the game sum of `CutExpand r` with itself and `CutExpand r` itself. -/
theorem cutExpand_fibration (r : α → α → Prop) :
    Fibration (GameAdd (CutExpand r) (CutExpand r)) (CutExpand r) fun s ↦ s.1 + s.2 := by
  /-
    α : Sort u_1
    x✝ : Sort u_2
    Fibration : x✝
    r : α → α → Prop
    ⊢ sorryAx (Sort u_3) Bool.true
  -/
  rintro ⟨s₁, s₂⟩ s ⟨t, a, hr, he⟩; dsimp at he ⊢
  classical
  obtain ⟨ha, rfl⟩ := add_singleton_eq_iff.1 he
  rw [add_assoc, mem_add] at ha
  obtain h | h := ha
  · refine ⟨(s₁.erase a + t, s₂), GameAdd.fst ⟨t, a, hr, ?_⟩, ?_⟩
    · rw [add_comm, ← add_assoc, singleton_add, cons_erase h]
    · rw [add_assoc s₁, erase_add_left_pos _ h, add_right_comm, add_assoc]
  · refine ⟨(s₁, (s₂ + t).erase a), GameAdd.snd ⟨t, a, hr, ?_⟩, ?_⟩
    · rw [add_comm, singleton_add, cons_erase h]
    · rw [add_assoc, erase_add_right_pos _ h]


-- Hi
COMPILATION STEP CONTENTS:
 /-- `CutExpand` preserves leftward-closedness under a relation. -/
lemma cutExpand_closed [IsIrrefl α r] (p : α → Prop)
    (h : ∀ {a' a}, r a' a → p a → p a') :
    ∀ {s' s}, CutExpand r s' s → (∀ a ∈ s, p a) → ∀ a ∈ s', p a := by
  intros s' s
  classical
  rw [cutExpand_iff]
  rintro ⟨t, a, hr, ha, rfl⟩ hsp a' h'
  obtain (h'|h') := mem_add.1 h'
  exacts [hsp a' (mem_of_mem_erase h'), h (hr a' h') (hsp a ha)]


AFTER ELAB CONTENTS:
 /-- `CutExpand` preserves leftward-closedness under a relation. -/
lemma cutExpand_closed [IsIrrefl α r] (p : α → Prop)
    (h : ∀ {a' a}, r a' a → p a → p a') :
    ∀ {s' s}, CutExpand r s' s → (∀ a ∈ s, p a) → ∀ a ∈ s', p a := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝ : IsIrrefl α r
    p : α → Prop
    h : ∀ {a' a : α}, r a' a → p a → p a'
    ⊢ ∀ {s' : ?m.643 p h} {s : ?m.644 p h}, sorryAx (Sort u_3) Bool.true → (∀ (a : …
  -/
  intros s' s
  classical
  rw [cutExpand_iff]
  rintro ⟨t, a, hr, ha, rfl⟩ hsp a' h'
  obtain (h'|h') := mem_add.1 h'
  exacts [hsp a' (mem_of_mem_erase h'), h (hr a' h') (hsp a ha)]


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


AFTER ELAB CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


AFTER ELAB CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


AFTER ELAB CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


AFTER ELAB CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


AFTER ELAB CONTENTS:
 lemma cutExpand_double {a a₁ a₂} (h₁ : r a₁ a) (h₂ : r a₂ a) : CutExpand r {a₁, a₂} {a} :=
  cutExpand_singleton <| by
    simp only [insert_eq_cons, mem_cons, mem_singleton, forall_eq_or_imp, forall_eq]
    tauto


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_pair_left {a' a b} (hr : r a' a) : CutExpand r {a', b} {a, b} :=
  (cutExpand_add_right {b}).2 (cutExpand_singleton_singleton hr)


AFTER ELAB CONTENTS:
 lemma cutExpand_pair_left {a' a b} (hr : r a' a) : CutExpand r {a', b} {a, b} :=
  (cutExpand_add_right {b}).2 (cutExpand_singleton_singleton hr)


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_pair_right {a b' b} (hr : r b' b) : CutExpand r {a, b'} {a, b} :=
  (cutExpand_add_left {a}).2 (cutExpand_singleton_singleton hr)


AFTER ELAB CONTENTS:
 lemma cutExpand_pair_right {a b' b} (hr : r b' b) : CutExpand r {a, b'} {a, b} :=
  (cutExpand_add_left {a}).2 (cutExpand_singleton_singleton hr)


-- Hi
COMPILATION STEP CONTENTS:
 lemma cutExpand_double_left {a a₁ a₂ b} (h₁ : r a₁ a) (h₂ : r a₂ a) :
    CutExpand r {a₁, a₂, b} {a, b} :=
  (cutExpand_add_right {b}).2 (cutExpand_double h₁ h₂)


AFTER ELAB CONTENTS:
 lemma cutExpand_double_left {a a₁ a₂ b} (h₁ : r a₁ a) (h₂ : r a₂ a) :
    CutExpand r {a₁, a₂, b} {a, b} :=
  (cutExpand_add_right {b}).2 (cutExpand_double h₁ h₂)


-- Hi
COMPILATION STEP CONTENTS:
 /-- A multiset is accessible under `CutExpand` if all its singleton subsets are,
  assuming `r` is irreflexive. -/
theorem acc_of_singleton [IsIrrefl α r] {s : Multiset α} (hs : ∀ a ∈ s, Acc (CutExpand r) {a}) :
    Acc (CutExpand r) s := by
  induction s using Multiset.induction with
  | empty => exact Acc.intro 0 fun s h ↦ (not_cutExpand_zero s h).elim
  | cons a s ihs =>
    rw [← s.singleton_add a]
    rw [forall_mem_cons] at hs
    exact (hs.1.prod_gameAdd <| ihs fun a ha ↦ hs.2 a ha).of_fibration _ (cutExpand_fibration r)


AFTER ELAB CONTENTS:
 /-- A multiset is accessible under `CutExpand` if all its singleton subsets are,
  assuming `r` is irreflexive. -/
theorem acc_of_singleton [IsIrrefl α r] {s : Multiset α} (hs : ∀ a ∈ s, Acc (CutExpand r) {a}) :
    Acc (CutExpand r) s := by
  induction s using Multiset.induction with
  | empty => exact Acc.intro 0 fun s h ↦ (not_cutExpand_zero s h).elim
  | cons a s ihs =>
    rw [← s.singleton_add a]
    rw [forall_mem_cons] at hs
    exact (hs.1.prod_gameAdd <| ihs fun a ha ↦ hs.2 a ha).of_fibration _ (cutExpand_fibration r)


-- Hi
COMPILATION STEP CONTENTS:
 /-- A singleton `{a}` is accessible under `CutExpand r` if `a` is accessible under `r`,
  assuming `r` is irreflexive. -/
theorem _root_.Acc.cutExpand [IsIrrefl α r] {a : α} (hacc : Acc r a) : Acc (CutExpand r) {a} := by
  induction' hacc with a h ih
  refine Acc.intro _ fun s ↦ ?_
  classical
  simp only [cutExpand_iff, mem_singleton]
  rintro ⟨t, a, hr, rfl, rfl⟩
  refine acc_of_singleton fun a' ↦ ?_
  rw [erase_singleton, zero_add]
  exact ih a' ∘ hr a'


AFTER ELAB CONTENTS:
 /-- A singleton `{a}` is accessible under `CutExpand r` if `a` is accessible under `r`,
  assuming `r` is irreflexive. -/
theorem _root_.Acc.cutExpand [IsIrrefl α r] {a : α} (hacc : Acc r a) : Acc (CutExpand r) {a} := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝ : IsIrrefl α r
    a : α
    hacc : Acc r a
    ⊢ Acc (sorryAx (?m.487 hacc → ?m.487 hacc → Prop) Bool.true) (Singleton.single …
  -/
  induction' hacc with a h ih
  /-
    case intro
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝ : IsIrrefl α r
    a✝ a : α
    h : ∀ (y : α), r y a → Acc r y
    ih : ∀ (y : α) (a : r y a), Acc (sorryAx (?m.487 ⋯ → ?m.487 ⋯ → Prop) Bool.tru …
    ⊢ Acc (sorryAx (?m.487 ⋯ → ?m.487 ⋯ → Prop) Bool.true) (Singleton.singleton a)
  -/
  refine Acc.intro _ fun s ↦ ?_
  classical
  simp only [cutExpand_iff, mem_singleton]
  rintro ⟨t, a, hr, rfl, rfl⟩
  refine acc_of_singleton fun a' ↦ ?_
  rw [erase_singleton, zero_add]
  exact ih a' ∘ hr a'


-- Hi
COMPILATION STEP CONTENTS:
 /-- A singleton `{a}` is accessible under `CutExpand r` if `a` is accessible under `r`,
  assuming `r` is irreflexive. -/
theorem _root_.Acc.cutExpand [IsIrrefl α r] {a : α} (hacc : Acc r a) : Acc (CutExpand r) {a} := by
  induction' hacc with a h ih
  refine Acc.intro _ fun s ↦ ?_
  classical
  simp only [cutExpand_iff, mem_singleton]
  rintro ⟨t, a, hr, rfl, rfl⟩
  refine acc_of_singleton fun a' ↦ ?_
  rw [erase_singleton, zero_add]
  exact ih a' ∘ hr a'


AFTER ELAB CONTENTS:
 /-- A singleton `{a}` is accessible under `CutExpand r` if `a` is accessible under `r`,
  assuming `r` is irreflexive. -/
theorem _root_.Acc.cutExpand [IsIrrefl α r] {a : α} (hacc : Acc r a) : Acc (CutExpand r) {a} := by
  /-
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝ : IsIrrefl α r
    a : α
    hacc : Acc r a
    ⊢ Acc (sorryAx (?m.487 hacc → ?m.487 hacc → Prop) Bool.true) (Singleton.single …
  -/
  induction' hacc with a h ih
  /-
    case intro
    α : Type u_1
    r : α → α → Prop
    x✝ : Sort u_2
    CutExpand : x✝
    inst✝ : IsIrrefl α r
    a✝ a : α
    h : ∀ (y : α), r y a → Acc r y
    ih : ∀ (y : α) (a : r y a), Acc (sorryAx (?m.487 ⋯ → ?m.487 ⋯ → Prop) Bool.tru …
    ⊢ Acc (sorryAx (?m.487 ⋯ → ?m.487 ⋯ → Prop) Bool.true) (Singleton.singleton a)
  -/
  refine Acc.intro _ fun s ↦ ?_
  classical
  simp only [cutExpand_iff, mem_singleton]
  rintro ⟨t, a, hr, rfl, rfl⟩
  refine acc_of_singleton fun a' ↦ ?_
  rw [erase_singleton, zero_add]
  exact ih a' ∘ hr a'


-- Hi
COMPILATION STEP CONTENTS:
 /-- `CutExpand r` is well-founded when `r` is. -/
theorem _root_.WellFounded.cutExpand (hr : WellFounded r) : WellFounded (CutExpand r) :=
  ⟨have := hr.isIrrefl; fun _ ↦ acc_of_singleton fun a _ ↦ (hr.apply a).cutExpand⟩


AFTER ELAB CONTENTS:
 /-- `CutExpand r` is well-founded when `r` is. -/
theorem _root_.WellFounded.cutExpand (hr : WellFounded r) : WellFounded (CutExpand r) :=
  ⟨have := hr.isIrrefl; fun _ ↦ acc_of_singleton fun a _ ↦ (hr.apply a).cutExpand⟩


-- Hi

-/
#eval runAtDecls `Mathlib.Logic.Hydra
