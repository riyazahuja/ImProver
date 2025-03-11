/-- The definition for an impartial game, defined using Conway induction. -/
def ImpartialAux (G : PGame) : Prop :=
  (G ≈ -G) ∧ (∀ i, ImpartialAux (G.moveLeft i)) ∧ ∀ j, ImpartialAux (G.moveRight j)
termination_by G


theorem impartialAux_def {G : PGame} : G.ImpartialAux ↔
    (G ≈ -G) ∧ (∀ i, ImpartialAux (G.moveLeft i)) ∧ ∀ j, ImpartialAux (G.moveRight j) := by
  /-
    G : SetTheory.PGame
    ⊢ Iff G.ImpartialAux (And (HasEquiv.Equiv G (Neg.neg G)) (And (∀ (i : G.LeftMo …
  -/
  rw [ImpartialAux]
  /-
    🎉 no goals
  -/


/-- A typeclass on impartial games. -/
class Impartial (G : PGame) : Prop where
  out : ImpartialAux G


theorem impartial_iff_aux {G : PGame} : G.Impartial ↔ G.ImpartialAux :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


theorem impartial_def {G : PGame} :
    G.Impartial ↔ (G ≈ -G) ∧ (∀ i, Impartial (G.moveLeft i)) ∧ ∀ j, Impartial (G.moveRight j) := by
  /-
    G : SetTheory.PGame
    ⊢ Iff G.Impartial (And (HasEquiv.Equiv G (Neg.neg G)) (And (∀ (i : G.LeftMoves …
  -/
  simpa only [impartial_iff_aux] using impartialAux_def
  /-
    🎉 no goals
  -/


instance impartial_zero : Impartial 0 := by
  /-
    ⊢ SetTheory.PGame.Impartial 0
  -/
  rw [impartial_def]
  /-
    ⊢ And (HasEquiv.Equiv 0 (-0)) (And (∀ (i : SetTheory.PGame.LeftMoves 0), (SetT …
  -/
  simp
  /-
    🎉 no goals
  -/


instance impartial_star : Impartial star := by
  /-
    ⊢ SetTheory.PGame.star.Impartial
  -/
  rw [impartial_def]
  /-
    ⊢ And (HasEquiv.Equiv SetTheory.PGame.star (Neg.neg SetTheory.PGame.star)) (An …
  -/
  simpa using Impartial.impartial_zero
  /-
    🎉 no goals
  -/


theorem neg_equiv_self (G : PGame) [h : G.Impartial] : G ≈ -G :=
  (impartial_def.1 h).1


@[simp]
theorem mk'_neg_equiv_self (G : PGame) [G.Impartial] : -(⟦G⟧ : Game) = ⟦G⟧ :=
  game_eq (Equiv.symm (neg_equiv_self G))


instance moveLeft_impartial {G : PGame} [h : G.Impartial] (i : G.LeftMoves) :
    (G.moveLeft i).Impartial :=
  (impartial_def.1 h).2.1 i


instance moveRight_impartial {G : PGame} [h : G.Impartial] (j : G.RightMoves) :
    (G.moveRight j).Impartial :=
  (impartial_def.1 h).2.2 j


theorem impartial_congr {G H : PGame} (e : G ≡r H) [G.Impartial] : H.Impartial :=
  impartial_def.2
    ⟨Equiv.trans e.symm.equiv (Equiv.trans (neg_equiv_self G) (neg_equiv_neg_iff.2 e.equiv)),
      fun i => impartial_congr (e.moveLeftSymm i), fun j => impartial_congr (e.moveRightSymm j)⟩
termination_by G


instance impartial_add (G H : PGame) [G.Impartial] [H.Impartial] : (G + H).Impartial := by
  /-
    G H : SetTheory.PGame
    inst✝¹ : G.Impartial
    inst✝ : H.Impartial
    ⊢ (HAdd.hAdd G H).Impartial
  -/
  rw [impartial_def]
  refine ⟨Equiv.trans (add_congr (neg_equiv_self G) (neg_equiv_self _))
      (Equiv.symm (negAddRelabelling _ _).equiv), fun k => ?_, fun k => ?_⟩
    /-
      case refine_1
      G H : SetTheory.PGame
      inst✝¹ : G.Impartial
      inst✝ : H.Impartial
      k : (HAdd.hAdd G H).LeftMoves
      ⊢ ((HAdd.hAdd G H).moveLeft k).Impartial
    -/
  · apply leftMoves_add_cases k
    all_goals
      intro i; simp only [add_moveLeft_inl, add_moveLeft_inr]
      apply impartial_add
    /-
      case refine_2
      G H : SetTheory.PGame
      inst✝¹ : G.Impartial
      inst✝ : H.Impartial
      k : (HAdd.hAdd G H).RightMoves
      ⊢ ((HAdd.hAdd G H).moveRight k).Impartial
    -/
  · apply rightMoves_add_cases k
    all_goals
      intro i; simp only [add_moveRight_inl, add_moveRight_inr]
      apply impartial_add
termination_by (G, H)


instance impartial_neg (G : PGame) [G.Impartial] : (-G).Impartial := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ (Neg.neg G).Impartial
  -/
  rw [impartial_def]
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ And (HasEquiv.Equiv (Neg.neg G) (Neg.neg (Neg.neg G))) (And (∀ (i : (Neg.neg …
  -/
  refine ⟨?_, fun i => ?_, fun i => ?_⟩
    /-
      case refine_1
      G : SetTheory.PGame
      inst✝ : G.Impartial
      ⊢ HasEquiv.Equiv (Neg.neg G) (Neg.neg (Neg.neg G))
    -/
  · rw [neg_neg]
    /-
      case refine_1
      G : SetTheory.PGame
      inst✝ : G.Impartial
      ⊢ HasEquiv.Equiv (Neg.neg G) G
    -/
    exact Equiv.symm (neg_equiv_self G)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : SetTheory.PGame
      inst✝ : G.Impartial
      i : (Neg.neg G).LeftMoves
      ⊢ ((Neg.neg G).moveLeft i).Impartial
    -/
  · rw [moveLeft_neg]
    /-
      case refine_2
      G : SetTheory.PGame
      inst✝ : G.Impartial
      i : (Neg.neg G).LeftMoves
      ⊢ (Neg.neg (G.moveRight (SetTheory.PGame.toLeftMovesNeg.symm i))).Impartial
    -/
    exact impartial_neg _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      G : SetTheory.PGame
      inst✝ : G.Impartial
      i : (Neg.neg G).RightMoves
      ⊢ ((Neg.neg G).moveRight i).Impartial
    -/
  · rw [moveRight_neg]
    /-
      case refine_3
      G : SetTheory.PGame
      inst✝ : G.Impartial
      i : (Neg.neg G).RightMoves
      ⊢ (Neg.neg (G.moveLeft (SetTheory.PGame.toRightMovesNeg.symm i))).Impartial
    -/
    exact impartial_neg _
    /-
      🎉 no goals
    -/
termination_by G


theorem nonpos : ¬0 < G := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Not (LT.lt 0 G)
  -/
  intro h
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    h : LT.lt 0 G
    ⊢ False
  -/
  have h' := neg_lt_neg_iff.2 h
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    h : LT.lt 0 G
    h' : LT.lt (Neg.neg G) (-0)
    ⊢ False
  -/
  rw [neg_zero, lt_congr_left (Equiv.symm (neg_equiv_self G))] at h'
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    h : LT.lt 0 G
    h' : LT.lt G 0
    ⊢ False
  -/
  exact (h.trans h').false
  /-
    🎉 no goals
  -/


theorem nonneg : ¬G < 0 := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Not (LT.lt G 0)
  -/
  intro h
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    h : LT.lt G 0
    ⊢ False
  -/
  have h' := neg_lt_neg_iff.2 h
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    h : LT.lt G 0
    h' : LT.lt (-0) (Neg.neg G)
    ⊢ False
  -/
  rw [neg_zero, lt_congr_right (Equiv.symm (neg_equiv_self G))] at h'
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    h : LT.lt G 0
    h' : LT.lt 0 G
    ⊢ False
  -/
  exact (h.trans h').false
  /-
    🎉 no goals
  -/


/-- In an impartial game, either the first player always wins, or the second player always wins. -/
theorem equiv_or_fuzzy_zero : (G ≈ 0) ∨ G ‖ 0 := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Or (HasEquiv.Equiv G 0) (G.Fuzzy 0)
  -/
  rcases lt_or_equiv_or_gt_or_fuzzy G 0 with (h | h | h | h)
    /-
      case inl
      G : SetTheory.PGame
      inst✝ : G.Impartial
      h : LT.lt G 0
      ⊢ Or (HasEquiv.Equiv G 0) (G.Fuzzy 0)
    -/
  · exact ((nonneg G) h).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      G : SetTheory.PGame
      inst✝ : G.Impartial
      h : HasEquiv.Equiv G 0
      ⊢ Or (HasEquiv.Equiv G 0) (G.Fuzzy 0)
    -/
  · exact Or.inl h
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inl
      G : SetTheory.PGame
      inst✝ : G.Impartial
      h : LT.lt 0 G
      ⊢ Or (HasEquiv.Equiv G 0) (G.Fuzzy 0)
    -/
  · exact ((nonpos G) h).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr
      G : SetTheory.PGame
      inst✝ : G.Impartial
      h : G.Fuzzy 0
      ⊢ Or (HasEquiv.Equiv G 0) (G.Fuzzy 0)
    -/
  · exact Or.inr h
    /-
      🎉 no goals
    -/


@[simp]
theorem not_equiv_zero_iff : ¬(G ≈ 0) ↔ G ‖ 0 :=
  ⟨(equiv_or_fuzzy_zero G).resolve_left, Fuzzy.not_equiv⟩


@[simp]
theorem not_fuzzy_zero_iff : ¬G ‖ 0 ↔ (G ≈ 0) :=
  ⟨(equiv_or_fuzzy_zero G).resolve_right, Equiv.not_fuzzy⟩


theorem add_self : G + G ≈ 0 :=
  Equiv.trans (add_congr_left (neg_equiv_self G)) (neg_add_cancel_equiv G)


@[simp]
theorem mk'_add_self : (⟦G⟧ : Game) + ⟦G⟧ = 0 :=
  game_eq (add_self G)


/-- This lemma doesn't require `H` to be impartial. -/
theorem equiv_iff_add_equiv_zero (H : PGame) : (H ≈ G) ↔ (H + G ≈ 0) := by
  rw [equiv_iff_game_eq, ← add_right_cancel_iff (a := ⟦G⟧), mk'_add_self, ← quot_add,
    equiv_iff_game_eq, quot_zero]


/-- This lemma doesn't require `H` to be impartial. -/
theorem equiv_iff_add_equiv_zero' (H : PGame) : (G ≈ H) ↔ (G + H ≈ 0) := by
  rw [equiv_iff_game_eq, ← add_left_cancel_iff, mk'_add_self, ← quot_add, equiv_iff_game_eq,
    Eq.comm, quot_zero]


theorem le_zero_iff {G : PGame} [G.Impartial] : G ≤ 0 ↔ 0 ≤ G := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (LE.le G 0) (LE.le 0 G)
  -/
  rw [← zero_le_neg_iff, le_congr_right (neg_equiv_self G)]
  /-
    🎉 no goals
  -/


theorem lf_zero_iff {G : PGame} [G.Impartial] : G ⧏ 0 ↔ 0 ⧏ G := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (G.LF 0) (SetTheory.PGame.LF 0 G)
  -/
  rw [← zero_lf_neg_iff, lf_congr_right (neg_equiv_self G)]
  /-
    🎉 no goals
  -/


theorem equiv_zero_iff_le : (G ≈ 0) ↔ G ≤ 0 :=
  ⟨And.left, fun h => ⟨h, le_zero_iff.1 h⟩⟩


theorem fuzzy_zero_iff_lf : G ‖ 0 ↔ G ⧏ 0 :=
  ⟨And.left, fun h => ⟨h, lf_zero_iff.1 h⟩⟩


theorem equiv_zero_iff_ge : (G ≈ 0) ↔ 0 ≤ G :=
  ⟨And.right, fun h => ⟨le_zero_iff.2 h, h⟩⟩


theorem fuzzy_zero_iff_gf : G ‖ 0 ↔ 0 ⧏ G :=
  ⟨And.right, fun h => ⟨lf_zero_iff.2 h, h⟩⟩


theorem forall_leftMoves_fuzzy_iff_equiv_zero : (∀ i, G.moveLeft i ‖ 0) ↔ (G ≈ 0) := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (∀ (i : G.LeftMoves), (G.moveLeft i).Fuzzy 0) (HasEquiv.Equiv G 0)
  -/
  refine ⟨fun hb => ?_, fun hp i => ?_⟩
    /-
      case refine_1
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hb : ∀ (i : G.LeftMoves), (G.moveLeft i).Fuzzy 0
      ⊢ HasEquiv.Equiv G 0
    -/
  · rw [equiv_zero_iff_le G, le_zero_lf]
    /-
      case refine_1
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hb : ∀ (i : G.LeftMoves), (G.moveLeft i).Fuzzy 0
      ⊢ ∀ (i : G.LeftMoves), (G.moveLeft i).LF 0
    -/
    exact fun i => (hb i).1
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hp : HasEquiv.Equiv G 0
      i : G.LeftMoves
      ⊢ (G.moveLeft i).Fuzzy 0
    -/
  · rw [fuzzy_zero_iff_lf]
    /-
      case refine_2
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hp : HasEquiv.Equiv G 0
      i : G.LeftMoves
      ⊢ (G.moveLeft i).LF 0
    -/
    exact hp.1.moveLeft_lf i
    /-
      🎉 no goals
    -/


theorem forall_rightMoves_fuzzy_iff_equiv_zero : (∀ j, G.moveRight j ‖ 0) ↔ (G ≈ 0) := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (∀ (j : G.RightMoves), (G.moveRight j).Fuzzy 0) (HasEquiv.Equiv G 0)
  -/
  refine ⟨fun hb => ?_, fun hp i => ?_⟩
    /-
      case refine_1
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hb : ∀ (j : G.RightMoves), (G.moveRight j).Fuzzy 0
      ⊢ HasEquiv.Equiv G 0
    -/
  · rw [equiv_zero_iff_ge G, zero_le_lf]
    /-
      case refine_1
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hb : ∀ (j : G.RightMoves), (G.moveRight j).Fuzzy 0
      ⊢ ∀ (j : G.RightMoves), SetTheory.PGame.LF 0 (G.moveRight j)
    -/
    exact fun i => (hb i).2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hp : HasEquiv.Equiv G 0
      i : G.RightMoves
      ⊢ (G.moveRight i).Fuzzy 0
    -/
  · rw [fuzzy_zero_iff_gf]
    /-
      case refine_2
      G : SetTheory.PGame
      inst✝ : G.Impartial
      hp : HasEquiv.Equiv G 0
      i : G.RightMoves
      ⊢ SetTheory.PGame.LF 0 (G.moveRight i)
    -/
    exact hp.2.lf_moveRight i
    /-
      🎉 no goals
    -/


theorem exists_left_move_equiv_iff_fuzzy_zero : (∃ i, G.moveLeft i ≈ 0) ↔ G ‖ 0 := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (Exists fun i => HasEquiv.Equiv (G.moveLeft i) 0) (G.Fuzzy 0)
  -/
  refine ⟨fun ⟨i, hi⟩ => (fuzzy_zero_iff_gf G).2 (lf_of_le_moveLeft hi.2), fun hn => ?_⟩
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    hn : G.Fuzzy 0
    ⊢ Exists fun i => HasEquiv.Equiv (G.moveLeft i) 0
  -/
  rw [fuzzy_zero_iff_gf G, zero_lf_le] at hn
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    hn : Exists fun i => LE.le 0 (G.moveLeft i)
    ⊢ Exists fun i => HasEquiv.Equiv (G.moveLeft i) 0
  -/
  cases' hn with i hi
  /-
    case intro
    G : SetTheory.PGame
    inst✝ : G.Impartial
    i : G.LeftMoves
    hi : LE.le 0 (G.moveLeft i)
    ⊢ Exists fun i => HasEquiv.Equiv (G.moveLeft i) 0
  -/
  exact ⟨i, (equiv_zero_iff_ge _).2 hi⟩
  /-
    🎉 no goals
  -/


theorem exists_right_move_equiv_iff_fuzzy_zero : (∃ j, G.moveRight j ≈ 0) ↔ G ‖ 0 := by
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    ⊢ Iff (Exists fun j => HasEquiv.Equiv (G.moveRight j) 0) (G.Fuzzy 0)
  -/
  refine ⟨fun ⟨i, hi⟩ => (fuzzy_zero_iff_lf G).2 (lf_of_moveRight_le hi.1), fun hn => ?_⟩
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    hn : G.Fuzzy 0
    ⊢ Exists fun j => HasEquiv.Equiv (G.moveRight j) 0
  -/
  rw [fuzzy_zero_iff_lf G, lf_zero_le] at hn
  /-
    G : SetTheory.PGame
    inst✝ : G.Impartial
    hn : Exists fun j => LE.le (G.moveRight j) 0
    ⊢ Exists fun j => HasEquiv.Equiv (G.moveRight j) 0
  -/
  cases' hn with i hi
  /-
    case intro
    G : SetTheory.PGame
    inst✝ : G.Impartial
    i : G.RightMoves
    hi : LE.le (G.moveRight i) 0
    ⊢ Exists fun j => HasEquiv.Equiv (G.moveRight j) 0
  -/
  exact ⟨i, (equiv_zero_iff_le _).2 hi⟩
  /-
    🎉 no goals
  -/


