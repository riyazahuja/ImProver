/-- A pre-game is numeric if everything in the L set is less than everything in the R set,
and all the elements of L and R are also numeric. -/
def Numeric : PGame → Prop
  | ⟨_, _, L, R⟩ => (∀ i j, L i < R j) ∧ (∀ i, Numeric (L i)) ∧ ∀ j, Numeric (R j)


theorem numeric_def {x : PGame} :
    Numeric x ↔
      (∀ i j, x.moveLeft i < x.moveRight j) ∧
        (∀ i, Numeric (x.moveLeft i)) ∧ ∀ j, Numeric (x.moveRight j) := by
  /-
    x : SetTheory.PGame
    ⊢ Iff x.Numeric (And (∀ (i : x.LeftMoves) (j : x.RightMoves), LT.lt (x.moveLef …
  -/
  cases x; rfl
           /-
             🎉 no goals
           -/


theorem mk {x : PGame} (h₁ : ∀ i j, x.moveLeft i < x.moveRight j) (h₂ : ∀ i, Numeric (x.moveLeft i))
    (h₃ : ∀ j, Numeric (x.moveRight j)) : Numeric x :=
  numeric_def.2 ⟨h₁, h₂, h₃⟩


theorem left_lt_right {x : PGame} (o : Numeric x) (i : x.LeftMoves) (j : x.RightMoves) :
                                       /-
                                         x : SetTheory.PGame
                                         o : x.Numeric
                                         i : x.LeftMoves
                                         j : x.RightMoves
                                         ⊢ LT.lt (x.moveLeft i) (x.moveRight j)
                                       -/
    x.moveLeft i < x.moveRight j := by cases x; exact o.1 i j
                                                /-
                                                  🎉 no goals
                                                -/


theorem moveLeft {x : PGame} (o : Numeric x) (i : x.LeftMoves) : Numeric (x.moveLeft i) := by
  /-
    x : SetTheory.PGame
    o : x.Numeric
    i : x.LeftMoves
    ⊢ (x.moveLeft i).Numeric
  -/
  cases x; exact o.2.1 i
           /-
             🎉 no goals
           -/


theorem moveRight {x : PGame} (o : Numeric x) (j : x.RightMoves) : Numeric (x.moveRight j) := by
  /-
    x : SetTheory.PGame
    o : x.Numeric
    j : x.RightMoves
    ⊢ (x.moveRight j).Numeric
  -/
  cases x; exact o.2.2 j
           /-
             🎉 no goals
           -/


lemma isOption {x' x} (h : IsOption x' x) (hx : Numeric x) : Numeric x' := by
  /-
    x' x : SetTheory.PGame
    h : x'.IsOption x
    hx : x.Numeric
    ⊢ x'.Numeric
  -/
  cases h
    /-
      case moveLeft
      x : SetTheory.PGame
      hx : x.Numeric
      i✝ : x.LeftMoves
      ⊢ (x.moveLeft i✝).Numeric
    -/
  · apply hx.moveLeft
    /-
      🎉 no goals
    -/
    /-
      case moveRight
      x : SetTheory.PGame
      hx : x.Numeric
      i✝ : x.RightMoves
      ⊢ (x.moveRight i✝).Numeric
    -/
  · apply hx.moveRight
    /-
      🎉 no goals
    -/


@[elab_as_elim]
theorem numeric_rec {C : PGame → Prop}
    (H : ∀ (l r) (L : l → PGame) (R : r → PGame), (∀ i j, L i < R j) →
      (∀ i, Numeric (L i)) → (∀ i, Numeric (R i)) → (∀ i, C (L i)) → (∀ i, C (R i)) →
      C ⟨l, r, L, R⟩) :
    ∀ x, Numeric x → C x
  | ⟨_, _, _, _⟩, ⟨h, hl, hr⟩ =>
    H _ _ _ _ h hl hr (fun i => numeric_rec H _ (hl i)) fun i => numeric_rec H _ (hr i)


theorem Relabelling.numeric_imp {x y : PGame} (r : x ≡r y) (ox : Numeric x) : Numeric y := by
  /-
    x y : SetTheory.PGame
    r : x.Relabelling y
    ox : x.Numeric
    ⊢ y.Numeric
  -/
  induction' x using PGame.moveRecOn with x IHl IHr generalizing y
  /-
    case IH
    x : SetTheory.PGame
    IHl : ∀ (i : x.LeftMoves) {y : SetTheory.PGame}, (x.moveLeft i).Relabelling y  …
    IHr : ∀ (j : x.RightMoves) {y : SetTheory.PGame}, (x.moveRight j).Relabelling  …
    y : SetTheory.PGame
    r : x.Relabelling y
    ox : x.Numeric
    ⊢ y.Numeric
  -/
  apply Numeric.mk (fun i j => ?_) (fun i => ?_) fun j => ?_
    /-
      x : SetTheory.PGame
      IHl : ∀ (i : x.LeftMoves) {y : SetTheory.PGame}, (x.moveLeft i).Relabelling y  …
      IHr : ∀ (j : x.RightMoves) {y : SetTheory.PGame}, (x.moveRight j).Relabelling  …
      y : SetTheory.PGame
      r : x.Relabelling y
      ox : x.Numeric
      i : y.LeftMoves
      j : y.RightMoves
      ⊢ LT.lt (y.moveLeft i) (y.moveRight j)
    -/
  · rw [← lt_congr (r.moveLeftSymm i).equiv (r.moveRightSymm j).equiv]
    /-
      x : SetTheory.PGame
      IHl : ∀ (i : x.LeftMoves) {y : SetTheory.PGame}, (x.moveLeft i).Relabelling y  …
      IHr : ∀ (j : x.RightMoves) {y : SetTheory.PGame}, (x.moveRight j).Relabelling  …
      y : SetTheory.PGame
      r : x.Relabelling y
      ox : x.Numeric
      i : y.LeftMoves
      j : y.RightMoves
      ⊢ LT.lt (x.moveLeft (r.leftMovesEquiv.symm i)) (x.moveRight (r.rightMovesEquiv …
    -/
    apply ox.left_lt_right
    /-
      🎉 no goals
    -/
    /-
      x : SetTheory.PGame
      IHl : ∀ (i : x.LeftMoves) {y : SetTheory.PGame}, (x.moveLeft i).Relabelling y  …
      IHr : ∀ (j : x.RightMoves) {y : SetTheory.PGame}, (x.moveRight j).Relabelling  …
      y : SetTheory.PGame
      r : x.Relabelling y
      ox : x.Numeric
      i : y.LeftMoves
      ⊢ (y.moveLeft i).Numeric
    -/
  · exact IHl _ (r.moveLeftSymm i) (ox.moveLeft _)
    /-
      🎉 no goals
    -/
    /-
      x : SetTheory.PGame
      IHl : ∀ (i : x.LeftMoves) {y : SetTheory.PGame}, (x.moveLeft i).Relabelling y  …
      IHr : ∀ (j : x.RightMoves) {y : SetTheory.PGame}, (x.moveRight j).Relabelling  …
      y : SetTheory.PGame
      r : x.Relabelling y
      ox : x.Numeric
      j : y.RightMoves
      ⊢ (y.moveRight j).Numeric
    -/
  · exact IHr _ (r.moveRightSymm j) (ox.moveRight _)
    /-
      🎉 no goals
    -/


/-- Relabellings preserve being numeric. -/
theorem Relabelling.numeric_congr {x y : PGame} (r : x ≡r y) : Numeric x ↔ Numeric y :=
  ⟨r.numeric_imp, r.symm.numeric_imp⟩


theorem lf_asymm {x y : PGame} (ox : Numeric x) (oy : Numeric y) : x ⧏ y → ¬y ⧏ x := by
  refine numeric_rec (C := fun x => ∀ z (_oz : Numeric z), x ⧏ z → ¬z ⧏ x)
    (fun xl xr xL xR hx _oxl _oxr IHxl IHxr => ?_) x ox y oy
  /-
    x y : SetTheory.PGame
    ox : x.Numeric
    oy : y.Numeric
    xl xr : Type u_1
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    hx : ∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)
    _oxl : ∀ (i : xl), (xL i).Numeric
    _oxr : ∀ (i : xr), (xR i).Numeric
    IHxl : ∀ (i : xl), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
    IHxr : ∀ (i : xr), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
    ⊢ (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not (z.LF x)) (SetTh …
  -/
  refine numeric_rec fun yl yr yL yR hy oyl oyr _IHyl _IHyr => ?_
  /-
    x y : SetTheory.PGame
    ox : x.Numeric
    oy : y.Numeric
    xl xr : Type u_1
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    hx : ∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)
    _oxl : ∀ (i : xl), (xL i).Numeric
    _oxr : ∀ (i : xr), (xR i).Numeric
    IHxl : ∀ (i : xl), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
    IHxr : ∀ (i : xr), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
    yl yr : Type u_1
    yL : yl → SetTheory.PGame
    yR : yr → SetTheory.PGame
    hy : ∀ (i : yl) (j : yr), LT.lt (yL i) (yR j)
    oyl : ∀ (i : yl), (yL i).Numeric
    oyr : ∀ (i : yr), (yR i).Numeric
    _IHyl : ∀ (i : yl), (SetTheory.PGame.mk xl xr xL xR).LF (yL i) → Not ((yL i).L …
    _IHyr : ∀ (i : yr), (SetTheory.PGame.mk xl xr xL xR).LF (yR i) → Not ((yR i).L …
    ⊢ (SetTheory.PGame.mk xl xr xL xR).LF (SetTheory.PGame.mk yl yr yL yR) → Not ( …
  -/
  rw [mk_lf_mk, mk_lf_mk]; rintro (⟨i, h₁⟩ | ⟨j, h₁⟩) (⟨i, h₂⟩ | ⟨j, h₂⟩)
    /-
      case inl.intro.inl.intro
      x y : SetTheory.PGame
      ox : x.Numeric
      oy : y.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      hx : ∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)
      _oxl : ∀ (i : xl), (xL i).Numeric
      _oxr : ∀ (i : xr), (xR i).Numeric
      IHxl : ∀ (i : xl), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      IHxr : ∀ (i : xr), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      hy : ∀ (i : yl) (j : yr), LT.lt (yL i) (yR j)
      oyl : ∀ (i : yl), (yL i).Numeric
      oyr : ∀ (i : yr), (yR i).Numeric
      _IHyl : ∀ (i : yl), (SetTheory.PGame.mk xl xr xL xR).LF (yL i) → Not ((yL i).L …
      _IHyr : ∀ (i : yr), (SetTheory.PGame.mk xl xr xL xR).LF (yR i) → Not ((yR i).L …
      i✝ : yl
      h₁ : LE.le (SetTheory.PGame.mk xl xr xL xR) (yL i✝)
      i : xl
      h₂ : LE.le (SetTheory.PGame.mk yl yr yL yR) (xL i)
      ⊢ False
    -/
  · exact IHxl _ _ (oyl _) (h₁.moveLeft_lf _) (h₂.moveLeft_lf _)
    /-
      🎉 no goals
    -/
    /-
      case inl.intro.inr.intro
      x y : SetTheory.PGame
      ox : x.Numeric
      oy : y.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      hx : ∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)
      _oxl : ∀ (i : xl), (xL i).Numeric
      _oxr : ∀ (i : xr), (xR i).Numeric
      IHxl : ∀ (i : xl), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      IHxr : ∀ (i : xr), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      hy : ∀ (i : yl) (j : yr), LT.lt (yL i) (yR j)
      oyl : ∀ (i : yl), (yL i).Numeric
      oyr : ∀ (i : yr), (yR i).Numeric
      _IHyl : ∀ (i : yl), (SetTheory.PGame.mk xl xr xL xR).LF (yL i) → Not ((yL i).L …
      _IHyr : ∀ (i : yr), (SetTheory.PGame.mk xl xr xL xR).LF (yR i) → Not ((yR i).L …
      i : yl
      h₁ : LE.le (SetTheory.PGame.mk xl xr xL xR) (yL i)
      j : yr
      h₂ : LE.le (yR j) (SetTheory.PGame.mk xl xr xL xR)
      ⊢ False
    -/
  · exact (le_trans h₂ h₁).not_gf (lf_of_lt (hy _ _))
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inl.intro
      x y : SetTheory.PGame
      ox : x.Numeric
      oy : y.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      hx : ∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)
      _oxl : ∀ (i : xl), (xL i).Numeric
      _oxr : ∀ (i : xr), (xR i).Numeric
      IHxl : ∀ (i : xl), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      IHxr : ∀ (i : xr), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      hy : ∀ (i : yl) (j : yr), LT.lt (yL i) (yR j)
      oyl : ∀ (i : yl), (yL i).Numeric
      oyr : ∀ (i : yr), (yR i).Numeric
      _IHyl : ∀ (i : yl), (SetTheory.PGame.mk xl xr xL xR).LF (yL i) → Not ((yL i).L …
      _IHyr : ∀ (i : yr), (SetTheory.PGame.mk xl xr xL xR).LF (yR i) → Not ((yR i).L …
      j : xr
      h₁ : LE.le (xR j) (SetTheory.PGame.mk yl yr yL yR)
      i : xl
      h₂ : LE.le (SetTheory.PGame.mk yl yr yL yR) (xL i)
      ⊢ False
    -/
  · exact (le_trans h₁ h₂).not_gf (lf_of_lt (hx _ _))
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.inr.intro
      x y : SetTheory.PGame
      ox : x.Numeric
      oy : y.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      hx : ∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)
      _oxl : ∀ (i : xl), (xL i).Numeric
      _oxr : ∀ (i : xr), (xR i).Numeric
      IHxl : ∀ (i : xl), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      IHxr : ∀ (i : xr), (fun x => ∀ (z : SetTheory.PGame), z.Numeric → x.LF z → Not …
      yl yr : Type u_1
      yL : yl → SetTheory.PGame
      yR : yr → SetTheory.PGame
      hy : ∀ (i : yl) (j : yr), LT.lt (yL i) (yR j)
      oyl : ∀ (i : yl), (yL i).Numeric
      oyr : ∀ (i : yr), (yR i).Numeric
      _IHyl : ∀ (i : yl), (SetTheory.PGame.mk xl xr xL xR).LF (yL i) → Not ((yL i).L …
      _IHyr : ∀ (i : yr), (SetTheory.PGame.mk xl xr xL xR).LF (yR i) → Not ((yR i).L …
      j✝ : xr
      h₁ : LE.le (xR j✝) (SetTheory.PGame.mk yl yr yL yR)
      j : yr
      h₂ : LE.le (yR j) (SetTheory.PGame.mk xl xr xL xR)
      ⊢ False
    -/
  · exact IHxr _ _ (oyr _) (h₁.lf_moveRight _) (h₂.lf_moveRight _)
    /-
      🎉 no goals
    -/


theorem le_of_lf {x y : PGame} (h : x ⧏ y) (ox : Numeric x) (oy : Numeric y) : x ≤ y :=
  not_lf.1 (lf_asymm ox oy h)


alias LF.le := le_of_lf


theorem lt_of_lf {x y : PGame} (h : x ⧏ y) (ox : Numeric x) (oy : Numeric y) : x < y :=
  (lt_or_fuzzy_of_lf h).resolve_right (not_fuzzy_of_le (h.le ox oy))


alias LF.lt := lt_of_lf


theorem lf_iff_lt {x y : PGame} (ox : Numeric x) (oy : Numeric y) : x ⧏ y ↔ x < y :=
  ⟨fun h => h.lt ox oy, lf_of_lt⟩


/-- Definition of `x ≤ y` on numeric pre-games, in terms of `<` -/
theorem le_iff_forall_lt {x y : PGame} (ox : x.Numeric) (oy : y.Numeric) :
    x ≤ y ↔ (∀ i, x.moveLeft i < y) ∧ ∀ j, x < y.moveRight j := by
  /-
    x y : SetTheory.PGame
    ox : x.Numeric
    oy : y.Numeric
    ⊢ Iff (LE.le x y) (And (∀ (i : x.LeftMoves), LT.lt (x.moveLeft i) y) (∀ (j : y …
  -/
  refine le_iff_forall_lf.trans (and_congr ?_ ?_) <;>
      /-
        case refine_1
        x y : SetTheory.PGame
        ox : x.Numeric
        oy : y.Numeric
        ⊢ Iff (∀ (i : x.LeftMoves), (x.moveLeft i).LF y) (∀ (i : x.LeftMoves), LT.lt ( …
      -/
      refine forall_congr' fun i => lf_iff_lt ?_ ?_ <;>
    /-
      case refine_1.refine_1
      x y : SetTheory.PGame
      ox : x.Numeric
      oy : y.Numeric
      i : x.LeftMoves
      ⊢ (x.moveLeft i).Numeric
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    apply_rules [Numeric.moveLeft, Numeric.moveRight]
    /-
      🎉 no goals
    -/


/-- Definition of `x < y` on numeric pre-games, in terms of `≤` -/
theorem lt_iff_exists_le {x y : PGame} (ox : x.Numeric) (oy : y.Numeric) :
    x < y ↔ (∃ i, x ≤ y.moveLeft i) ∨ ∃ j, x.moveRight j ≤ y := by
  /-
    x y : SetTheory.PGame
    ox : x.Numeric
    oy : y.Numeric
    ⊢ Iff (LT.lt x y) (Or (Exists fun i => LE.le x (y.moveLeft i)) (Exists fun j = …
  -/
  rw [← lf_iff_lt ox oy, lf_iff_exists_le]
  /-
    🎉 no goals
  -/


theorem lt_of_exists_le {x y : PGame} (ox : x.Numeric) (oy : y.Numeric) :
    ((∃ i, x ≤ y.moveLeft i) ∨ ∃ j, x.moveRight j ≤ y) → x < y :=
  (lt_iff_exists_le ox oy).2


/-- The definition of `x < y` on numeric pre-games, in terms of `<` two moves later. -/
theorem lt_def {x y : PGame} (ox : x.Numeric) (oy : y.Numeric) :
    x < y ↔
      (∃ i, (∀ i', x.moveLeft i' < y.moveLeft i) ∧ ∀ j, x < (y.moveLeft i).moveRight j) ∨
        ∃ j, (∀ i, (x.moveRight j).moveLeft i < y) ∧ ∀ j', x.moveRight j < y.moveRight j' := by
  /-
    x y : SetTheory.PGame
    ox : x.Numeric
    oy : y.Numeric
    ⊢ Iff (LT.lt x y) (Or (Exists fun i => And (∀ (i' : x.LeftMoves), LT.lt (x.mov …
  -/
  rw [← lf_iff_lt ox oy, lf_def]
  /-
    x y : SetTheory.PGame
    ox : x.Numeric
    oy : y.Numeric
    ⊢ Iff (Or (Exists fun i => And (∀ (i' : x.LeftMoves), (x.moveLeft i').LF (y.mo …
  -/
  refine or_congr ?_ ?_ <;> refine exists_congr fun x_1 => ?_ <;> refine and_congr ?_ ?_ <;>
      /-
        case refine_1.refine_1
        x y : SetTheory.PGame
        ox : x.Numeric
        oy : y.Numeric
        x_1 : y.LeftMoves
        ⊢ Iff (∀ (i' : x.LeftMoves), (x.moveLeft i').LF (y.moveLeft x_1)) (∀ (i' : x.L …
      -/
      refine forall_congr' fun i => lf_iff_lt ?_ ?_ <;>
    /-
      case refine_1.refine_1.refine_1
      x y : SetTheory.PGame
      ox : x.Numeric
      oy : y.Numeric
      x_1 : y.LeftMoves
      i : x.LeftMoves
      ⊢ (x.moveLeft i).Numeric
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    apply_rules [Numeric.moveLeft, Numeric.moveRight]
    /-
      🎉 no goals
    -/


theorem not_fuzzy {x y : PGame} (ox : Numeric x) (oy : Numeric y) : ¬Fuzzy x y :=
  fun h => not_lf.2 ((lf_of_fuzzy h).le ox oy) h.2


theorem lt_or_equiv_or_gt {x y : PGame} (ox : Numeric x) (oy : Numeric y) :
    x < y ∨ (x ≈ y) ∨ y < x :=
  ((lf_or_equiv_or_gf x y).imp fun h => h.lt ox oy) <| Or.imp_right fun h => h.lt oy ox


theorem numeric_of_isEmpty (x : PGame) [IsEmpty x.LeftMoves] [IsEmpty x.RightMoves] : Numeric x :=
  Numeric.mk isEmptyElim isEmptyElim isEmptyElim


theorem numeric_of_isEmpty_leftMoves (x : PGame) [IsEmpty x.LeftMoves] :
    (∀ j, Numeric (x.moveRight j)) → Numeric x :=
  Numeric.mk isEmptyElim isEmptyElim


theorem numeric_of_isEmpty_rightMoves (x : PGame) [IsEmpty x.RightMoves]
    (H : ∀ i, Numeric (x.moveLeft i)) : Numeric x :=
  Numeric.mk (fun _ => isEmptyElim) H isEmptyElim


theorem numeric_zero : Numeric 0 :=
  numeric_of_isEmpty 0


theorem numeric_one : Numeric 1 :=
  numeric_of_isEmpty_rightMoves 1 fun _ => numeric_zero


theorem Numeric.neg : ∀ {x : PGame} (_ : Numeric x), Numeric (-x)
  | ⟨_, _, _, _⟩, o =>
    ⟨fun j i => neg_lt_neg_iff.2 (o.1 i j), fun j => (o.2.2 j).neg, fun i => (o.2.1 i).neg⟩


/-- Inserting a smaller numeric left option into a numeric game results in a numeric game. -/
theorem insertLeft_numeric {x x' : PGame} (x_num : x.Numeric) (x'_num : x'.Numeric)
    (h : x' ≤ x) : (insertLeft x x').Numeric := by
  /-
    x x' : SetTheory.PGame
    x_num : x.Numeric
    x'_num : x'.Numeric
    h : LE.le x' x
    ⊢ (x.insertLeft x').Numeric
  -/
  rw [le_iff_forall_lt x'_num x_num] at h
  /-
    x x' : SetTheory.PGame
    x_num : x.Numeric
    x'_num : x'.Numeric
    h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) x) (∀ (j : x.RightMoves), …
    ⊢ (x.insertLeft x').Numeric
  -/
  unfold Numeric at x_num ⊢
  /-
    x x' : SetTheory.PGame
    x'_num : x'.Numeric
    h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) x) (∀ (j : x.RightMoves), …
    x_num : SetTheory.PGame.Numeric.match_1 (fun x => Prop) x fun α β L R => And ( …
    ⊢ SetTheory.PGame.Numeric.match_1 (fun x => Prop) (x.insertLeft x') fun α β L  …
  -/
  rcases x with ⟨xl, xr, xL, xR⟩
  /-
    case mk
    x' : SetTheory.PGame
    x'_num : x'.Numeric
    xl xr : Type u_1
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) (SetTheory.PGame.mk xl xr …
    x_num : SetTheory.PGame.Numeric.match_1 (fun x => Prop) (SetTheory.PGame.mk xl …
    ⊢ SetTheory.PGame.Numeric.match_1 (fun x => Prop) ((SetTheory.PGame.mk xl xr x …
  -/
  simp only [insertLeft, Sum.forall, forall_const, Sum.elim_inl, Sum.elim_inr] at x_num ⊢
  /-
    case mk
    x' : SetTheory.PGame
    x'_num : x'.Numeric
    xl xr : Type u_1
    xL : xl → SetTheory.PGame
    xR : xr → SetTheory.PGame
    h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) (SetTheory.PGame.mk xl xr …
    x_num : And (∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)) (And (∀ (i : xl), (xL i …
    ⊢ And (And (∀ (a : xl) (j : xr), LT.lt (xL a) (xR j)) (∀ (j : xr), LT.lt x' (x …
  -/
  constructor
    /-
      case mk.left
      x' : SetTheory.PGame
      x'_num : x'.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) (SetTheory.PGame.mk xl xr …
      x_num : And (∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)) (And (∀ (i : xl), (xL i …
      ⊢ And (∀ (a : xl) (j : xr), LT.lt (xL a) (xR j)) (∀ (j : xr), LT.lt x' (xR j))
    -/
  · simp only [x_num.1, implies_true, true_and]
    /-
      case mk.left
      x' : SetTheory.PGame
      x'_num : x'.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) (SetTheory.PGame.mk xl xr …
      x_num : And (∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)) (And (∀ (i : xl), (xL i …
      ⊢ ∀ (j : xr), LT.lt x' (xR j)
    -/
    simp only [rightMoves_mk, moveRight_mk] at h
    /-
      case mk.left
      x' : SetTheory.PGame
      x'_num : x'.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) (SetTheory.PGame.mk xl xr …
      x_num : And (∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)) (And (∀ (i : xl), (xL i …
      ⊢ ∀ (j : xr), LT.lt x' (xR j)
    -/
    exact h.2
    /-
      🎉 no goals
    -/
    /-
      case mk.right
      x' : SetTheory.PGame
      x'_num : x'.Numeric
      xl xr : Type u_1
      xL : xl → SetTheory.PGame
      xR : xr → SetTheory.PGame
      h : And (∀ (i : x'.LeftMoves), LT.lt (x'.moveLeft i) (SetTheory.PGame.mk xl xr …
      x_num : And (∀ (i : xl) (j : xr), LT.lt (xL i) (xR j)) (And (∀ (i : xl), (xL i …
      ⊢ And (And (∀ (a : xl), (xL a).Numeric) x'.Numeric) (∀ (j : xr), (xR j).Numeric)
    -/
  · simp only [x_num, implies_true, x'_num, and_self]
    /-
      🎉 no goals
    -/


/-- Inserting a larger numeric right option into a numeric game results in a numeric game. -/
theorem insertRight_numeric {x x' : PGame} (x_num : x.Numeric) (x'_num : x'.Numeric)
    (h : x ≤ x') : (insertRight x x').Numeric := by
  /-
    x x' : SetTheory.PGame
    x_num : x.Numeric
    x'_num : x'.Numeric
    h : LE.le x x'
    ⊢ (x.insertRight x').Numeric
  -/
  rw [← neg_neg (x.insertRight x'), ← neg_insertLeft_neg]
  /-
    x x' : SetTheory.PGame
    x_num : x.Numeric
    x'_num : x'.Numeric
    h : LE.le x x'
    ⊢ (Neg.neg ((Neg.neg x).insertLeft (Neg.neg x'))).Numeric
  -/
  apply Numeric.neg
  /-
    case x
    x x' : SetTheory.PGame
    x_num : x.Numeric
    x'_num : x'.Numeric
    h : LE.le x x'
    ⊢ ((Neg.neg x).insertLeft (Neg.neg x')).Numeric
  -/
  exact insertLeft_numeric (Numeric.neg x_num) (Numeric.neg x'_num) (neg_le_neg_iff.mpr h)
  /-
    🎉 no goals
  -/


theorem moveLeft_lt {x : PGame} (o : Numeric x) (i) : x.moveLeft i < x :=
  (moveLeft_lf i).lt (o.moveLeft i) o


theorem moveLeft_le {x : PGame} (o : Numeric x) (i) : x.moveLeft i ≤ x :=
  (o.moveLeft_lt i).le


theorem lt_moveRight {x : PGame} (o : Numeric x) (j) : x < x.moveRight j :=
  (lf_moveRight j).lt o (o.moveRight j)


theorem le_moveRight {x : PGame} (o : Numeric x) (j) : x ≤ x.moveRight j :=
  (o.lt_moveRight j).le


theorem add : ∀ {x y : PGame} (_ : Numeric x) (_ : Numeric y), Numeric (x + y)
  | ⟨xl, xr, xL, xR⟩, ⟨yl, yr, yL, yR⟩, ox, oy =>
    ⟨by
      /-
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
        oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
        ⊢ ∀ (i : Sum xl yl) (j : Sum xr yr), LT.lt ((fun t => Sum.rec (fun i => (fun a …
      -/
      rintro (ix | iy) (jx | jy)
        /-
          case inl.inl
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
          oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
          ix : xl
          jx : xr
          ⊢ LT.lt ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.rec (motive :=  …
        -/
      · exact add_lt_add_right (ox.1 ix jx) _
        /-
          🎉 no goals
        -/
      · exact (add_lf_add_of_lf_of_le (lf_mk _ _ ix) (oy.le_moveRight jy)).lt
          ((ox.moveLeft ix).add oy) (ox.add (oy.moveRight jy))
      · exact (add_lf_add_of_lf_of_le (mk_lf _ _ jx) (oy.moveLeft_le iy)).lt
          (ox.add (oy.moveLeft iy)) ((ox.moveRight jx).add oy)
        /-
          case inr.inr
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
          oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
          iy : yl
          jy : yr
          ⊢ LT.lt ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.rec (motive :=  …
        -/
      · exact add_lt_add_left (oy.1 iy jy) ⟨xl, xr, xL, xR⟩, by
        /-
          🎉 no goals
        -/
      /-
        xl xr : Type u_1
        xL : xl → SetTheory.PGame
        xR : xr → SetTheory.PGame
        yl yr : Type u_1
        yL : yl → SetTheory.PGame
        yR : yr → SetTheory.PGame
        ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
        oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
        ⊢ And (∀ (i : Sum xl yl), ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGa …
      -/
      constructor
        /-
          case left
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
          oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
          ⊢ ∀ (i : Sum xl yl), ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.re …
        -/
      · rintro (ix | iy)
          /-
            case left.inl
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
            oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
            ix : xl
            ⊢ ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.rec (motive := fun x  …
          -/
        · exact (ox.moveLeft ix).add oy
          /-
            🎉 no goals
          -/
          /-
            case left.inr
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
            oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
            iy : yl
            ⊢ ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.rec (motive := fun x  …
          -/
        · exact ox.add (oy.moveLeft iy)
          /-
            🎉 no goals
          -/
        /-
          case right
          xl xr : Type u_1
          xL : xl → SetTheory.PGame
          xR : xr → SetTheory.PGame
          yl yr : Type u_1
          yL : yl → SetTheory.PGame
          yR : yr → SetTheory.PGame
          ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
          oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
          ⊢ ∀ (j : Sum xr yr), ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.re …
        -/
      · rintro (jx | jy)
          /-
            case right.inl
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
            oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
            jx : xr
            ⊢ ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.rec (motive := fun x  …
          -/
        · apply (ox.moveRight jx).add oy
          /-
            🎉 no goals
          -/
          /-
            case right.inr
            xl xr : Type u_1
            xL : xl → SetTheory.PGame
            xR : xr → SetTheory.PGame
            yl yr : Type u_1
            yL : yl → SetTheory.PGame
            yR : yr → SetTheory.PGame
            ox : (SetTheory.PGame.mk xl xr xL xR).Numeric
            oy : (SetTheory.PGame.mk yl yr yL yR).Numeric
            jy : yr
            ⊢ ((fun t => Sum.rec (fun i => (fun a => SetTheory.PGame.rec (motive := fun x  …
          -/
        · apply ox.add (oy.moveRight jy)⟩
          /-
            🎉 no goals
          -/
termination_by x y => (x, y) -- Porting note: Added `termination_by`


theorem sub {x y : PGame} (ox : Numeric x) (oy : Numeric y) : Numeric (x - y) :=
  ox.add oy.neg


/-- Pre-games defined by natural numbers are numeric. -/
theorem numeric_nat : ∀ n : ℕ, Numeric n
  | 0 => numeric_zero
  | n + 1 => (numeric_nat n).add numeric_one


/-- Ordinal games are numeric. -/
theorem numeric_toPGame (o : Ordinal) : o.toPGame.Numeric := by
  /-
    o : Ordinal.{u_1}
    ⊢ o.toPGame.Numeric
  -/
  induction' o using Ordinal.induction with o IH
  /-
    case h
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → k.toPGame.Numeric
    ⊢ o.toPGame.Numeric
  -/
  apply numeric_of_isEmpty_rightMoves
  /-
    case h.H
    o : Ordinal.{u_1}
    IH : ∀ (k : Ordinal.{u_1}), LT.lt k o → k.toPGame.Numeric
    ⊢ ∀ (i : o.toPGame.LeftMoves), (o.toPGame.moveLeft i).Numeric
  -/
  simpa using fun i => IH _ (Ordinal.toLeftMovesToPGame_symm_lt i)
  /-
    🎉 no goals
  -/


/-- The type of surreal numbers. These are the numeric pre-games quotiented
by the equivalence relation `x ≈ y ↔ x ≤ y ∧ y ≤ x`. In the quotient,
the order becomes a total order. -/
def Surreal :=
  Quotient (inferInstanceAs <| Setoid (Subtype Numeric))


/-- Construct a surreal number from a numeric pre-game. -/
def mk (x : PGame) (h : x.Numeric) : Surreal :=
  ⟦⟨x, h⟩⟧


instance : Zero Surreal :=
  ⟨mk 0 numeric_zero⟩


instance : One Surreal :=
  ⟨mk 1 numeric_one⟩


instance : Inhabited Surreal :=
  ⟨0⟩


lemma mk_eq_mk {x y : PGame.{u}} {hx hy} : mk x hx = mk y hy ↔ x ≈ y := Quotient.eq


lemma mk_eq_zero {x : PGame.{u}} {hx} : mk x hx = 0 ↔ x ≈ 0 := Quotient.eq


/-- Lift an equivalence-respecting function on pre-games to surreals. -/
def lift {α} (f : ∀ x, Numeric x → α)
    (H : ∀ {x y} (hx : Numeric x) (hy : Numeric y), x.Equiv y → f x hx = f y hy) : Surreal → α :=
  Quotient.lift (fun x : { x // Numeric x } => f x.1 x.2) fun x y => H x.2 y.2


/-- Lift a binary equivalence-respecting function on pre-games to surreals. -/
def lift₂ {α} (f : ∀ x y, Numeric x → Numeric y → α)
    (H :
      ∀ {x₁ y₁ x₂ y₂} (ox₁ : Numeric x₁) (oy₁ : Numeric y₁) (ox₂ : Numeric x₂) (oy₂ : Numeric y₂),
        x₁.Equiv x₂ → y₁.Equiv y₂ → f x₁ y₁ ox₁ oy₁ = f x₂ y₂ ox₂ oy₂) :
    Surreal → Surreal → α :=
  lift (fun x ox => lift (fun y oy => f x y ox oy) fun _ _ => H _ _ _ _ equiv_rfl)
    fun _ _ h => funext <| Quotient.ind fun _ => H _ _ _ _ h equiv_rfl


instance instLE : LE Surreal :=
  ⟨lift₂ (fun x y _ _ => x ≤ y) fun _ _ _ _ hx hy => propext (le_congr hx hy)⟩


@[simp]
lemma mk_le_mk {x y : PGame.{u}} {hx hy} : mk x hx ≤ mk y hy ↔ x ≤ y := Iff.rfl


lemma zero_le_mk {x : PGame.{u}} {hx} : 0 ≤ mk x hx ↔ 0 ≤ x := Iff.rfl


instance instLT : LT Surreal :=
  ⟨lift₂ (fun x y _ _ => x < y) fun _ _ _ _ hx hy => propext (lt_congr hx hy)⟩


lemma mk_lt_mk {x y : PGame.{u}} {hx hy} : mk x hx < mk y hy ↔ x < y := Iff.rfl


lemma zero_lt_mk {x : PGame.{u}} {hx} : 0 < mk x hx ↔ 0 < x := Iff.rfl


/-- Same as `moveLeft_lt`, but for `Surreal` instead of `PGame` -/
theorem mk_moveLeft_lt_mk {x : PGame} (o : Numeric x) (i) :
    Surreal.mk (x.moveLeft i) (Numeric.moveLeft o i) < Surreal.mk x o := Numeric.moveLeft_lt o i


/-- Same as `lt_moveRight`, but for `Surreal` instead of `PGame` -/
theorem mk_lt_mk_moveRight {x : PGame} (o : Numeric x) (j) :
    Surreal.mk x o < Surreal.mk (x.moveRight j) (Numeric.moveRight o j) := Numeric.lt_moveRight o j


/-- Addition on surreals is inherited from pre-game addition:
the sum of `x = {xL | xR}` and `y = {yL | yR}` is `{xL + y, x + yL | xR + y, x + yR}`. -/
instance : Add Surreal :=
  ⟨Surreal.lift₂ (fun (x y : PGame) ox oy => ⟦⟨x + y, ox.add oy⟩⟧) fun _ _ _ _ hx hy =>
      Quotient.sound (add_congr hx hy)⟩


/-- Negation for surreal numbers is inherited from pre-game negation:
the negation of `{L | R}` is `{-R | -L}`. -/
instance : Neg Surreal :=
  ⟨Surreal.lift (fun x ox => ⟦⟨-x, ox.neg⟩⟧) fun _ _ a => Quotient.sound (neg_equiv_neg_iff.2 a)⟩


instance orderedAddCommGroup : OrderedAddCommGroup Surreal where
  add := (· + ·)
                  /-
                    ⊢ ∀ (a b c : Surreal), Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAd …
                  -/
  add_assoc := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; exact Quotient.sound add_assoc_equiv
                                      /-
                                        🎉 no goals
                                      -/
  zero := 0
                 /-
                   ⊢ ∀ (a : Surreal), Eq (HAdd.hAdd 0 a) a
                 -/
  zero_add := by rintro ⟨a⟩; exact Quotient.sound (zero_add_equiv a)
                             /-
                               🎉 no goals
                             -/
                 /-
                   ⊢ ∀ (a : Surreal), Eq (HAdd.hAdd a 0) a
                 -/
  add_zero := by rintro ⟨a⟩; exact Quotient.sound (add_zero_equiv a)
                             /-
                               🎉 no goals
                             -/
  neg := Neg.neg
                       /-
                         ⊢ ∀ (a : Surreal), Eq (HAdd.hAdd (Neg.neg a) a) 0
                       -/
  neg_add_cancel := by rintro ⟨a⟩; exact Quotient.sound (neg_add_cancel_equiv a)
                                   /-
                                     🎉 no goals
                                   -/
                 /-
                   ⊢ ∀ (a b : Surreal), Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                 -/
  add_comm := by rintro ⟨_⟩ ⟨_⟩; exact Quotient.sound add_comm_equiv
                                 /-
                                   🎉 no goals
                                 -/
  le := (· ≤ ·)
  lt := (· < ·)
                /-
                  ⊢ ∀ (a : Surreal), LE.le a a
                -/
  le_refl := by rintro ⟨_⟩; apply @le_rfl PGame
                            /-
                              🎉 no goals
                            -/
                 /-
                   ⊢ ∀ (a b c : Surreal), LE.le a b → LE.le b c → LE.le a c
                 -/
  le_trans := by rintro ⟨_⟩ ⟨_⟩ ⟨_⟩; apply @le_trans PGame
                                     /-
                                       🎉 no goals
                                     -/
                         /-
                           ⊢ ∀ (a b : Surreal), Iff (LT.lt a b) (And (LE.le a b) (Not (LE.le b a)))
                         -/
  lt_iff_le_not_le := by rintro ⟨_, ox⟩ ⟨_, oy⟩; apply @lt_iff_le_not_le PGame
                                                 /-
                                                   🎉 no goals
                                                 -/
                    /-
                      ⊢ ∀ (a b : Surreal), LE.le a b → LE.le b a → Eq a b
                    -/
  le_antisymm := by rintro ⟨_⟩ ⟨_⟩ h₁ h₂; exact Quotient.sound ⟨h₁, h₂⟩
                                          /-
                                            🎉 no goals
                                          -/
                        /-
                          ⊢ ∀ (a b : Surreal), LE.le a b → ∀ (c : Surreal), LE.le (HAdd.hAdd c a) (HAdd. …
                        -/
  add_le_add_left := by rintro ⟨_⟩ ⟨_⟩ hx ⟨_⟩; exact @add_le_add_left PGame _ _ _ _ _ hx _
                                               /-
                                                 🎉 no goals
                                               -/
  nsmul := nsmulRec
  zsmul := zsmulRec


lemma mk_add {x y : PGame} (hx : x.Numeric) (hy : y.Numeric) :
                                                                             /-
                                                                               x y : SetTheory.PGame
                                                                               hx : x.Numeric
                                                                               hy : y.Numeric
                                                                               ⊢ Eq (Surreal.mk (HAdd.hAdd x y) ⋯) (HAdd.hAdd (Surreal.mk x hx) (Surreal.mk y …
                                                                             -/
    Surreal.mk (x + y) (hx.add hy) = Surreal.mk x hx + Surreal.mk y hy := by rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


lemma mk_sub {x y : PGame} (hx : x.Numeric) (hy : y.Numeric) :
                                                                             /-
                                                                               x y : SetTheory.PGame
                                                                               hx : x.Numeric
                                                                               hy : y.Numeric
                                                                               ⊢ Eq (Surreal.mk (HSub.hSub x y) ⋯) (HSub.hSub (Surreal.mk x hx) (Surreal.mk y …
                                                                             -/
    Surreal.mk (x - y) (hx.sub hy) = Surreal.mk x hx - Surreal.mk y hy := by rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


                                             /-
                                               ⊢ Eq 0 (Surreal.mk 0 SetTheory.PGame.numeric_zero)
                                             -/
lemma zero_def : 0 = mk 0 numeric_zero := by rfl
                                             /-
                                               🎉 no goals
                                             -/


noncomputable instance : LinearOrderedAddCommGroup Surreal :=
  { Surreal.orderedAddCommGroup with
    le_total := by
      /-
        ⊢ ∀ (a b : Surreal), Or (LE.le a b) (LE.le b a)
      -/
      rintro ⟨⟨x, ox⟩⟩ ⟨⟨y, oy⟩⟩
      /-
        case mk.mk.mk.mk
        a✝ : Surreal
        x : SetTheory.PGame
        ox : x.Numeric
        b✝ : Surreal
        y : SetTheory.PGame
        oy : y.Numeric
        ⊢ Or (LE.le (Quot.mk ⇑(inferInstanceAs (Setoid (Subtype SetTheory.PGame.Numeri …
      -/
      exact or_iff_not_imp_left.2 fun h => (PGame.not_le.1 h).le oy ox
      /-
        🎉 no goals
      -/
    decidableLE := Classical.decRel _ }


instance : AddMonoidWithOne Surreal :=
  AddMonoidWithOne.unary


/-- Casts a `Surreal` number into a `Game`. -/
def toGame : Surreal →+o Game where
  toFun := lift (fun x _ => ⟦x⟧) fun _ _ => Quot.sound
  map_zero' := rfl
                 /-
                   ⊢ ∀ (x y : Surreal), Eq ({ toFun := Surreal.lift (fun x x_1 => Quotient.mk Set …
                 -/
  map_add' := by rintro ⟨_, _⟩ ⟨_, _⟩; rfl
                                       /-
                                         🎉 no goals
                                       -/
                  /-
                    ⊢ Monotone (↑{ toFun := Surreal.lift (fun x x_1 => Quotient.mk SetTheory.PGame …
                  -/
  monotone' := by rintro ⟨_, _⟩ ⟨_, _⟩; exact id
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem zero_toGame : toGame 0 = 0 :=
  rfl


@[simp]
theorem one_toGame : toGame 1 = 1 :=
  rfl


@[simp]
theorem nat_toGame : ∀ n : ℕ, toGame n = n :=
  map_natCast' _ one_toGame


/-- A small family of surreals is bounded above. -/
lemma bddAbove_range_of_small {ι : Type*} [Small.{u} ι] (f : ι → Surreal.{u}) :
    BddAbove (Set.range f) := by
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Surreal
    ⊢ BddAbove (Set.range f)
  -/
  induction' f using Quotient.induction_on_pi with f
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    ⊢ BddAbove (Set.range fun i => Quotient.mk (inferInstanceAs (Setoid (Subtype S …
  -/
  let g : ι → PGame.{u} := Subtype.val ∘ f
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    g : ι → SetTheory.PGame := Function.comp Subtype.val f
    ⊢ BddAbove (Set.range fun i => Quotient.mk (inferInstanceAs (Setoid (Subtype S …
  -/
  have hg (i) : (g i).Numeric := Subtype.prop _
  conv in (⟦f _⟧) =>
    change mk (g i) (hg i)
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    g : ι → SetTheory.PGame := Function.comp Subtype.val f
    hg : ∀ (i : ι), (g i).Numeric
    ⊢ BddAbove (Set.range fun i => Surreal.mk (g i) ⋯)
  -/
  clear_value g
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    g : ι → SetTheory.PGame
    hg : ∀ (i : ι), (g i).Numeric
    ⊢ BddAbove (Set.range fun i => Surreal.mk (g i) ⋯)
  -/
  clear f
  let x : PGame.{u} := ⟨Σ i, (g <| (equivShrink.{u} ι).symm i).LeftMoves, PEmpty,
    fun x ↦ moveLeft _ x.2, PEmpty.elim⟩
  refine ⟨mk x (.mk (by simp [x]) (fun _ ↦ (hg _).moveLeft _) (by simp [x])),
    Set.forall_mem_range.2 fun i ↦ ?_⟩
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    g : ι → SetTheory.PGame
    hg : ∀ (i : ι), (g i).Numeric
    x : SetTheory.PGame := SetTheory.PGame.mk (Sigma fun i => (g ((equivShrink ι). …
    i : ι
    ⊢ LE.le (Surreal.mk (g i) ⋯) (Surreal.mk x ⋯)
  -/
  rw [mk_le_mk, ← (equivShrink ι).symm_apply_apply i, le_iff_forall_lf]
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    g : ι → SetTheory.PGame
    hg : ∀ (i : ι), (g i).Numeric
    x : SetTheory.PGame := SetTheory.PGame.mk (Sigma fun i => (g ((equivShrink ι). …
    i : ι
    ⊢ And (∀ (i_1 : (g ((equivShrink ι).symm ((equivShrink ι) i))).LeftMoves), ((g …
  -/
  simpa [x] using fun j ↦ @moveLeft_lf x ⟨equivShrink ι i, j⟩
  /-
    🎉 no goals
  -/


/-- A small set of surreals is bounded above. -/
lemma bddAbove_of_small (s : Set Surreal.{u}) [Small.{u} s] : BddAbove s := by
  /-
    s : Set Surreal
    inst✝ : Small.{u, u + 1} ↑s
    ⊢ BddAbove s
  -/
  simpa using bddAbove_range_of_small (Subtype.val : s → Surreal.{u})
  /-
    🎉 no goals
  -/


/-- A small family of surreals is bounded below. -/
lemma bddBelow_range_of_small {ι : Type*} [Small.{u} ι] (f : ι → Surreal.{u}) :
    BddBelow (Set.range f) := by
  /-
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Surreal
    ⊢ BddBelow (Set.range f)
  -/
  induction' f using Quotient.induction_on_pi with f
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    ⊢ BddBelow (Set.range fun i => Quotient.mk (inferInstanceAs (Setoid (Subtype S …
  -/
  let g : ι → PGame.{u} := Subtype.val ∘ f
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    g : ι → SetTheory.PGame := Function.comp Subtype.val f
    ⊢ BddBelow (Set.range fun i => Quotient.mk (inferInstanceAs (Setoid (Subtype S …
  -/
  have hg (i) : (g i).Numeric := Subtype.prop _
  conv in (⟦f _⟧) =>
    change mk (g i) (hg i)
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    g : ι → SetTheory.PGame := Function.comp Subtype.val f
    hg : ∀ (i : ι), (g i).Numeric
    ⊢ BddBelow (Set.range fun i => Surreal.mk (g i) ⋯)
  -/
  clear_value g
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    f : ι → Subtype SetTheory.PGame.Numeric
    g : ι → SetTheory.PGame
    hg : ∀ (i : ι), (g i).Numeric
    ⊢ BddBelow (Set.range fun i => Surreal.mk (g i) ⋯)
  -/
  clear f
  let x : PGame.{u} := ⟨PEmpty, Σ i, (g <| (equivShrink.{u} ι).symm i).RightMoves,
    PEmpty.elim, fun x ↦ moveRight _ x.2⟩
  refine ⟨mk x (.mk (by simp [x]) (by simp [x]) (fun _ ↦ (hg _).moveRight _) ),
    Set.forall_mem_range.2 fun i ↦ ?_⟩
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    g : ι → SetTheory.PGame
    hg : ∀ (i : ι), (g i).Numeric
    x : SetTheory.PGame := SetTheory.PGame.mk PEmpty.{u + 1} (Sigma fun i => (g (( …
    i : ι
    ⊢ LE.le (Surreal.mk x ⋯) (Surreal.mk (g i) ⋯)
  -/
  rw [mk_le_mk, ← (equivShrink ι).symm_apply_apply i, le_iff_forall_lf]
  /-
    case h
    ι : Type u_1
    inst✝ : Small.{u, u_1} ι
    g : ι → SetTheory.PGame
    hg : ∀ (i : ι), (g i).Numeric
    x : SetTheory.PGame := SetTheory.PGame.mk PEmpty.{u + 1} (Sigma fun i => (g (( …
    i : ι
    ⊢ And (∀ (i_1 : x.LeftMoves), (x.moveLeft i_1).LF (g ((equivShrink ι).symm ((e …
  -/
  simpa [x] using fun j ↦ @lf_moveRight x ⟨equivShrink ι i, j⟩
  /-
    🎉 no goals
  -/


/-- A small set of surreals is bounded below. -/
lemma bddBelow_of_small (s : Set Surreal.{u}) [Small.{u} s] : BddBelow s := by
  /-
    s : Set Surreal
    inst✝ : Small.{u, u + 1} ↑s
    ⊢ BddBelow s
  -/
  simpa using bddBelow_range_of_small (Subtype.val : s → Surreal.{u})
  /-
    🎉 no goals
  -/


/-- Converts an ordinal into the corresponding surreal. -/
noncomputable def toSurreal : Ordinal ↪o Surreal where
  toFun o := mk _ (numeric_toPGame o)
                                        /-
                                          a b : Ordinal.{?u.84966}
                                          h : Eq ((fun o => Surreal.mk o.toPGame ⋯) a) ((fun o => Surreal.mk o.toPGame ⋯ …
                                          ⊢ HasEquiv.Equiv a.toPGame b.toPGame
                                        -/
  inj' a b h := toPGame_equiv_iff.1 (by apply Quotient.exact h) -- Porting note: Added `by apply`
                                        /-
                                          🎉 no goals
                                        -/
  map_rel_iff' := @toPGame_le_iff


