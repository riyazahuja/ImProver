/-- The equivalence `(x, y) ↦ (x, y+1)`. -/
@[simps!]
def shiftUp : ℤ × ℤ ≃ ℤ × ℤ :=
  (Equiv.refl ℤ).prodCongr (Equiv.addRight (1 : ℤ))


/-- The equivalence `(x, y) ↦ (x+1, y)`. -/
@[simps!]
def shiftRight : ℤ × ℤ ≃ ℤ × ℤ :=
  (Equiv.addRight (1 : ℤ)).prodCongr (Equiv.refl ℤ)


/-- A Domineering board is an arbitrary finite subset of `ℤ × ℤ`. -/
-- Porting note: reducibility cannot be `local`. For now there are no dependents of this file so
-- being globally reducible is fine.
abbrev Board :=
  Finset (ℤ × ℤ)


/-- Left can play anywhere that a square and the square below it are open. -/
def left (b : Board) : Finset (ℤ × ℤ) :=
  b ∩ b.map shiftUp


/-- Right can play anywhere that a square and the square to the left are open. -/
def right (b : Board) : Finset (ℤ × ℤ) :=
  b ∩ b.map shiftRight


theorem mem_left {b : Board} (x : ℤ × ℤ) : x ∈ left b ↔ x ∈ b ∧ (x.1, x.2 - 1) ∈ b :=
  Finset.mem_inter.trans (and_congr Iff.rfl Finset.mem_map_equiv)


theorem mem_right {b : Board} (x : ℤ × ℤ) : x ∈ right b ↔ x ∈ b ∧ (x.1 - 1, x.2) ∈ b :=
  Finset.mem_inter.trans (and_congr Iff.rfl Finset.mem_map_equiv)


/-- After Left moves, two vertically adjacent squares are removed from the board. -/
def moveLeft (b : Board) (m : ℤ × ℤ) : Board :=
  (b.erase m).erase (m.1, m.2 - 1)


/-- After Left moves, two horizontally adjacent squares are removed from the board. -/
def moveRight (b : Board) (m : ℤ × ℤ) : Board :=
  (b.erase m).erase (m.1 - 1, m.2)


theorem fst_pred_mem_erase_of_mem_right {b : Board} {m : ℤ × ℤ} (h : m ∈ right b) :
    (m.1 - 1, m.2) ∈ b.erase m := by
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    ⊢ Membership.mem (Finset.erase b m) { fst := HSub.hSub m.1 1, snd := m.2 }
  -/
  rw [mem_right] at h
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : And (Membership.mem b m) (Membership.mem b { fst := HSub.hSub m.1 1, snd : …
    ⊢ Membership.mem (Finset.erase b m) { fst := HSub.hSub m.1 1, snd := m.2 }
  -/
  apply Finset.mem_erase_of_ne_of_mem _ h.2
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : And (Membership.mem b m) (Membership.mem b { fst := HSub.hSub m.1 1, snd : …
    ⊢ Ne { fst := HSub.hSub m.1 1, snd := m.2 } m
  -/
  exact ne_of_apply_ne Prod.fst (pred_ne_self m.1)
  /-
    🎉 no goals
  -/


theorem snd_pred_mem_erase_of_mem_left {b : Board} {m : ℤ × ℤ} (h : m ∈ left b) :
    (m.1, m.2 - 1) ∈ b.erase m := by
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    ⊢ Membership.mem (Finset.erase b m) { fst := m.1, snd := HSub.hSub m.2 1 }
  -/
  rw [mem_left] at h
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : And (Membership.mem b m) (Membership.mem b { fst := m.1, snd := HSub.hSub  …
    ⊢ Membership.mem (Finset.erase b m) { fst := m.1, snd := HSub.hSub m.2 1 }
  -/
  apply Finset.mem_erase_of_ne_of_mem _ h.2
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : And (Membership.mem b m) (Membership.mem b { fst := m.1, snd := HSub.hSub  …
    ⊢ Ne { fst := m.1, snd := HSub.hSub m.2 1 } m
  -/
  exact ne_of_apply_ne Prod.snd (pred_ne_self m.2)
  /-
    🎉 no goals
  -/


theorem card_of_mem_left {b : Board} {m : ℤ × ℤ} (h : m ∈ left b) : 2 ≤ Finset.card b := by
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    ⊢ LE.le 2 (Finset.card b)
  -/
  have w₁ : m ∈ b := (Finset.mem_inter.1 h).1
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    w₁ : Membership.mem b m
    ⊢ LE.le 2 (Finset.card b)
  -/
  have w₂ : (m.1, m.2 - 1) ∈ b.erase m := snd_pred_mem_erase_of_mem_left h
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    w₁ : Membership.mem b m
    w₂ : Membership.mem (Finset.erase b m) { fst := m.1, snd := HSub.hSub m.2 1 }
    ⊢ LE.le 2 (Finset.card b)
  -/
  have i₁ := Finset.card_erase_lt_of_mem w₁
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    w₁ : Membership.mem b m
    w₂ : Membership.mem (Finset.erase b m) { fst := m.1, snd := HSub.hSub m.2 1 }
    i₁ : LT.lt (Finset.erase b m).card (Finset.card b)
    ⊢ LE.le 2 (Finset.card b)
  -/
  have i₂ := Nat.lt_of_le_of_lt (Nat.zero_le _) (Finset.card_erase_lt_of_mem w₂)
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    w₁ : Membership.mem b m
    w₂ : Membership.mem (Finset.erase b m) { fst := m.1, snd := HSub.hSub m.2 1 }
    i₁ : LT.lt (Finset.erase b m).card (Finset.card b)
    i₂ : LT.lt 0 (Finset.erase b m).card
    ⊢ LE.le 2 (Finset.card b)
  -/
  exact Nat.lt_of_le_of_lt i₂ i₁
  /-
    🎉 no goals
  -/


theorem card_of_mem_right {b : Board} {m : ℤ × ℤ} (h : m ∈ right b) : 2 ≤ Finset.card b := by
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    ⊢ LE.le 2 (Finset.card b)
  -/
  have w₁ : m ∈ b := (Finset.mem_inter.1 h).1
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    w₁ : Membership.mem b m
    ⊢ LE.le 2 (Finset.card b)
  -/
  have w₂ := fst_pred_mem_erase_of_mem_right h
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    w₁ : Membership.mem b m
    w₂ : Membership.mem (Finset.erase b m) { fst := HSub.hSub m.1 1, snd := m.2 }
    ⊢ LE.le 2 (Finset.card b)
  -/
  have i₁ := Finset.card_erase_lt_of_mem w₁
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    w₁ : Membership.mem b m
    w₂ : Membership.mem (Finset.erase b m) { fst := HSub.hSub m.1 1, snd := m.2 }
    i₁ : LT.lt (Finset.erase b m).card (Finset.card b)
    ⊢ LE.le 2 (Finset.card b)
  -/
  have i₂ := Nat.lt_of_le_of_lt (Nat.zero_le _) (Finset.card_erase_lt_of_mem w₂)
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    w₁ : Membership.mem b m
    w₂ : Membership.mem (Finset.erase b m) { fst := HSub.hSub m.1 1, snd := m.2 }
    i₁ : LT.lt (Finset.erase b m).card (Finset.card b)
    i₂ : LT.lt 0 (Finset.erase b m).card
    ⊢ LE.le 2 (Finset.card b)
  -/
  exact Nat.lt_of_le_of_lt i₂ i₁
  /-
    🎉 no goals
  -/


theorem moveLeft_card {b : Board} {m : ℤ × ℤ} (h : m ∈ left b) :
    Finset.card (moveLeft b m) + 2 = Finset.card b := by
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    ⊢ Eq (HAdd.hAdd (Finset.card (SetTheory.PGame.Domineering.moveLeft b m)) 2) (F …
  -/
  dsimp only [moveLeft]
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    ⊢ Eq (HAdd.hAdd ((Finset.erase b m).erase { fst := m.1, snd := HSub.hSub m.2 1 …
  -/
  rw [Finset.card_erase_of_mem (snd_pred_mem_erase_of_mem_left h)]
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    ⊢ Eq (HAdd.hAdd (HSub.hSub (Finset.erase b m).card 1) 2) (Finset.card b)
  -/
  rw [Finset.card_erase_of_mem (Finset.mem_of_mem_inter_left h)]
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.left b) m
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (Finset.card b) 1) 1) 2) (Finset.card b)
  -/
  exact tsub_add_cancel_of_le (card_of_mem_left h)
  /-
    🎉 no goals
  -/


theorem moveRight_card {b : Board} {m : ℤ × ℤ} (h : m ∈ right b) :
    Finset.card (moveRight b m) + 2 = Finset.card b := by
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    ⊢ Eq (HAdd.hAdd (Finset.card (SetTheory.PGame.Domineering.moveRight b m)) 2) ( …
  -/
  dsimp only [moveRight]
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    ⊢ Eq (HAdd.hAdd ((Finset.erase b m).erase { fst := HSub.hSub m.1 1, snd := m.2 …
  -/
  rw [Finset.card_erase_of_mem (fst_pred_mem_erase_of_mem_right h)]
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    ⊢ Eq (HAdd.hAdd (HSub.hSub (Finset.erase b m).card 1) 2) (Finset.card b)
  -/
  rw [Finset.card_erase_of_mem (Finset.mem_of_mem_inter_left h)]
  /-
    b : SetTheory.PGame.Domineering.Board
    m : Prod Int Int
    h : Membership.mem (SetTheory.PGame.Domineering.right b) m
    ⊢ Eq (HAdd.hAdd (HSub.hSub (HSub.hSub (Finset.card b) 1) 1) 2) (Finset.card b)
  -/
  exact tsub_add_cancel_of_le (card_of_mem_right h)
  /-
    🎉 no goals
  -/


theorem moveLeft_smaller {b : Board} {m : ℤ × ℤ} (h : m ∈ left b) :
                                                             /-
                                                               b : SetTheory.PGame.Domineering.Board
                                                               m : Prod Int Int
                                                               h : Membership.mem (SetTheory.PGame.Domineering.left b) m
                                                               ⊢ LT.lt (HDiv.hDiv (Finset.card (SetTheory.PGame.Domineering.moveLeft b m)) 2) …
                                                             -/
    Finset.card (moveLeft b m) / 2 < Finset.card b / 2 := by simp [← moveLeft_card h, lt_add_one]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem moveRight_smaller {b : Board} {m : ℤ × ℤ} (h : m ∈ right b) :
                                                              /-
                                                                b : SetTheory.PGame.Domineering.Board
                                                                m : Prod Int Int
                                                                h : Membership.mem (SetTheory.PGame.Domineering.right b) m
                                                                ⊢ LT.lt (HDiv.hDiv (Finset.card (SetTheory.PGame.Domineering.moveRight b m)) 2 …
                                                              -/
    Finset.card (moveRight b m) / 2 < Finset.card b / 2 := by simp [← moveRight_card h, lt_add_one]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- The instance describing allowed moves on a Domineering board. -/
instance state : State Board where
  turnBound s := s.card / 2
  l s := (left s).image (moveLeft s)
  r s := (right s).image (moveRight s)
  left_bound m := by
    /-
      s✝ t✝ : SetTheory.PGame.Domineering.Board
      m : Membership.mem ((fun s => Finset.image (SetTheory.PGame.Domineering.moveLe …
      ⊢ LT.lt ((fun s => HDiv.hDiv (Finset.card s) 2) t✝) ((fun s => HDiv.hDiv (Fins …
    -/
    simp only [Finset.mem_image, Prod.exists] at m
    /-
      s✝ t✝ : SetTheory.PGame.Domineering.Board
      m : Exists fun a => Exists fun b => And (Membership.mem (SetTheory.PGame.Domin …
      ⊢ LT.lt ((fun s => HDiv.hDiv (Finset.card s) 2) t✝) ((fun s => HDiv.hDiv (Fins …
    -/
    rcases m with ⟨_, _, ⟨h, rfl⟩⟩
    /-
      case intro.intro.intro
      s✝ : SetTheory.PGame.Domineering.Board
      w✝¹ w✝ : Int
      h : Membership.mem (SetTheory.PGame.Domineering.left s✝) { fst := w✝¹, snd :=  …
      ⊢ LT.lt ((fun s => HDiv.hDiv (Finset.card s) 2) (SetTheory.PGame.Domineering.m …
    -/
    exact moveLeft_smaller h
    /-
      🎉 no goals
    -/
  right_bound m := by
    /-
      s✝ t✝ : SetTheory.PGame.Domineering.Board
      m : Membership.mem ((fun s => Finset.image (SetTheory.PGame.Domineering.moveRi …
      ⊢ LT.lt ((fun s => HDiv.hDiv (Finset.card s) 2) t✝) ((fun s => HDiv.hDiv (Fins …
    -/
    simp only [Finset.mem_image, Prod.exists] at m
    /-
      s✝ t✝ : SetTheory.PGame.Domineering.Board
      m : Exists fun a => Exists fun b => And (Membership.mem (SetTheory.PGame.Domin …
      ⊢ LT.lt ((fun s => HDiv.hDiv (Finset.card s) 2) t✝) ((fun s => HDiv.hDiv (Fins …
    -/
    rcases m with ⟨_, _, ⟨h, rfl⟩⟩
    /-
      case intro.intro.intro
      s✝ : SetTheory.PGame.Domineering.Board
      w✝¹ w✝ : Int
      h : Membership.mem (SetTheory.PGame.Domineering.right s✝) { fst := w✝¹, snd := …
      ⊢ LT.lt ((fun s => HDiv.hDiv (Finset.card s) 2) (SetTheory.PGame.Domineering.m …
    -/
    exact moveRight_smaller h
    /-
      🎉 no goals
    -/


/-- Construct a pre-game from a Domineering board. -/
def domineering (b : Domineering.Board) : PGame :=
  PGame.ofState b


/-- All games of Domineering are short, because each move removes two squares. -/
instance shortDomineering (b : Domineering.Board) : Short (domineering b) := by
  /-
    b : SetTheory.PGame.Domineering.Board
    ⊢ (SetTheory.PGame.domineering b).Short
  -/
  dsimp only [domineering]
  /-
    b : SetTheory.PGame.Domineering.Board
    ⊢ (SetTheory.PGame.ofState b).Short
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The Domineering board with two squares arranged vertically, in which Left has the only move. -/
def domineering.one :=
  domineering [(0, 0), (0, 1)].toFinset


/-- The `L` shaped Domineering board, in which Left is exactly half a move ahead. -/
def domineering.L :=
  domineering [(0, 2), (0, 1), (0, 0), (1, 0)].toFinset


                                                /-
                                                  ⊢ SetTheory.PGame.domineering.one.Short
                                                -/
instance shortOne : Short domineering.one := by dsimp [domineering.one]; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


                                            /-
                                              ⊢ SetTheory.PGame.domineering.L.Short
                                            -/
instance shortL : Short domineering.L := by dsimp [domineering.L]; infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/

-- The VM can play small games successfully:
-- #eval decide (domineering.one ≈ 1)
-- #eval decide (domineering.L + domineering.L ≈ 1)
-- The following no longer works since Lean 3.29, since definitions by well-founded
-- recursion no longer reduce definitionally.
-- We can check that `Decidable` instances reduce as expected,
-- and so our implementation of domineering is computable.
-- run_cmd tactic.whnf `(by apply_instance : Decidable (domineering.one ≤ 1)) >>= tactic.trace
-- dec_trivial can handle most of the dictionary of small games described in [conway2001]
-- example : domineering.one ≈ 1 := by decide
-- example : domineering.L + domineering.L ≈ 1 := by decide
-- example : domineering.L ≈ PGame.ofLists [0] [1] := by decide
-- example : (domineering ([(0,0), (0,1), (0,2), (0,3)].toFinset) ≈ 2) := by decide
-- example : (domineering ([(0,0), (0,1), (1,0), (1,1)].toFinset) ≈ PGame.ofLists [1] [-1]) :=
--   by decide
-- The 3x3 grid is doable, but takes a minute...
-- example :
--   (domineering ([(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)].toFinset) ≈
--     PGame.ofLists [1] [-1]) := by decide
-- The 5x5 grid is actually 0, but brute-forcing this is too challenging even for the VM.
-- #eval decide (domineering ([
--   (0,0), (0,1), (0,2), (0,3), (0,4),
--   (1,0), (1,1), (1,2), (1,3), (1,4),
--   (2,0), (2,1), (2,2), (2,3), (2,4),
--   (3,0), (3,1), (3,2), (3,3), (3,4),
--   (4,0), (4,1), (4,2), (4,3), (4,4)
--   ].toFinset) ≈ 0)

