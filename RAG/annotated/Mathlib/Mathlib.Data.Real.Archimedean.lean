instance instArchimedean : Archimedean ℝ :=
  archimedean_iff_rat_le.2 fun x =>
    Real.ind_mk x fun f =>
      let ⟨M, _, H⟩ := f.bounded' 0
      ⟨M, mk_le_of_forall_le ⟨0, fun i _ => Rat.cast_le.2 <| le_of_lt (abs_lt.1 (H i)).2⟩⟩


noncomputable instance : FloorRing ℝ :=
  Archimedean.floorRing _


theorem isCauSeq_iff_lift {f : ℕ → ℚ} : IsCauSeq abs f ↔ IsCauSeq abs fun i => (f i : ℝ) where
  mp H ε ε0 :=
    let ⟨δ, δ0, δε⟩ := exists_pos_rat_lt ε0
                                     /-
                                       f : Nat → Rat
                                       H : IsCauSeq abs f
                                       ε : Real
                                       ε0 : GT.gt ε 0
                                       δ : Rat
                                       δ0 : LT.lt 0 δ
                                       δε : LT.lt (↑δ) ε
                                       i : Nat
                                       hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub (f j) (f i))) δ
                                       j : Nat
                                       ij : GE.ge j i
                                       ⊢ LT.lt (abs (HSub.hSub ((fun i => ↑(f i)) j) ((fun i => ↑(f i)) i))) ε
                                     -/
    (H _ δ0).imp fun i hi j ij => by dsimp; exact lt_trans (mod_cast hi _ ij) δε
                                            /-
                                              🎉 no goals
                                            -/
  mpr H ε ε0 :=
                                                      /-
                                                        f : Nat → Rat
                                                        H : IsCauSeq abs fun i => ↑(f i)
                                                        ε : Rat
                                                        ε0 : GT.gt ε 0
                                                        i : Nat
                                                        hi : ∀ (j : Nat), GE.ge j i → LT.lt (abs (HSub.hSub ((fun i => ↑(f i)) j) ((fu …
                                                        j : Nat
                                                        ij : GE.ge j i
                                                        ⊢ LT.lt (abs (HSub.hSub (f j) (f i))) ε
                                                      -/
    (H _ (Rat.cast_pos.2 ε0)).imp fun i hi j ij => by dsimp at hi; exact mod_cast hi _ ij
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem of_near (f : ℕ → ℚ) (x : ℝ) (h : ∀ ε > 0, ∃ i, ∀ j ≥ i, |(f j : ℝ) - x| < ε) :
    ∃ h', Real.mk ⟨f, h'⟩ = x :=
  ⟨isCauSeq_iff_lift.2 (CauSeq.of_near _ (const abs x) h),
    sub_eq_zero.1 <|
      abs_eq_zero.1 <|
        (eq_of_le_of_forall_le_of_dense (abs_nonneg _)) fun _ε ε0 =>
          mk_near_of_forall_near <| (h _ ε0).imp fun _i h j ij => le_of_lt (h j ij)⟩


theorem exists_floor (x : ℝ) : ∃ ub : ℤ, (ub : ℝ) ≤ x ∧ ∀ z : ℤ, (z : ℝ) ≤ x → z ≤ ub :=
  Int.exists_greatest_of_bdd
    (let ⟨n, hn⟩ := exists_int_gt x
    ⟨n, fun _ h' => Int.cast_le.1 <| le_trans h' <| le_of_lt hn⟩)
    (let ⟨n, hn⟩ := exists_int_lt x
    ⟨n, le_of_lt hn⟩)


theorem exists_isLUB (hne : s.Nonempty) (hbdd : BddAbove s) : ∃ x, IsLUB s x := by
  /-
    s : Set Real
    hne : s.Nonempty
    hbdd : BddAbove s
    ⊢ Exists fun x => IsLUB s x
  -/
  rcases hne, hbdd with ⟨⟨L, hL⟩, ⟨U, hU⟩⟩
  have : ∀ d : ℕ, BddAbove { m : ℤ | ∃ y ∈ s, (m : ℝ) ≤ y * d } := by
    cases' exists_int_gt U with k hk
    refine fun d => ⟨k * d, fun z h => ?_⟩
    rcases h with ⟨y, yS, hy⟩
    refine Int.cast_le.1 (hy.trans ?_)
    push_cast
    exact mul_le_mul_of_nonneg_right ((hU yS).trans hk.le) d.cast_nonneg
  choose f hf using fun d : ℕ =>
    Int.exists_greatest_of_bdd (this d) ⟨⌊L * d⌋, L, hL, Int.floor_le _⟩
  have hf₁ : ∀ n > 0, ∃ y ∈ s, ((f n / n : ℚ) : ℝ) ≤ y := fun n n0 =>
    let ⟨y, yS, hy⟩ := (hf n).1
    ⟨y, yS, by simpa using (div_le_iff₀ (Nat.cast_pos.2 n0 : (_ : ℝ) < _)).2 hy⟩
  have hf₂ : ∀ n > 0, ∀ y ∈ s, (y - ((n : ℕ) : ℝ)⁻¹) < (f n / n : ℚ) := by
    intro n n0 y yS
    have := (Int.sub_one_lt_floor _).trans_le (Int.cast_le.2 <| (hf n).2 _ ⟨y, yS, Int.floor_le _⟩)
    simp only [Rat.cast_div, Rat.cast_intCast, Rat.cast_natCast, gt_iff_lt]
    rwa [lt_div_iff₀ (Nat.cast_pos.2 n0 : (_ : ℝ) < _), sub_mul, inv_mul_cancel₀]
    exact ne_of_gt (Nat.cast_pos.2 n0)
  have hg : IsCauSeq abs (fun n => f n / n : ℕ → ℚ) := by
    intro ε ε0
    suffices ∀ j ≥ ⌈ε⁻¹⌉₊, ∀ k ≥ ⌈ε⁻¹⌉₊, (f j / j - f k / k : ℚ) < ε by
      refine ⟨_, fun j ij => abs_lt.2 ⟨?_, this _ ij _ le_rfl⟩⟩
      rw [neg_lt, neg_sub]
      exact this _ le_rfl _ ij
    intro j ij k ik
    replace ij := le_trans (Nat.le_ceil _) (Nat.cast_le.2 ij)
    replace ik := le_trans (Nat.le_ceil _) (Nat.cast_le.2 ik)
    have j0 := Nat.cast_pos.1 ((inv_pos.2 ε0).trans_le ij)
    have k0 := Nat.cast_pos.1 ((inv_pos.2 ε0).trans_le ik)
    rcases hf₁ _ j0 with ⟨y, yS, hy⟩
    refine lt_of_lt_of_le ((Rat.cast_lt (K := ℝ)).1 ?_) ((inv_le_comm₀ ε0 (Nat.cast_pos.2 k0)).1 ik)
    simpa using sub_lt_iff_lt_add'.2 (lt_of_le_of_lt hy <| sub_lt_iff_lt_add.1 <| hf₂ _ k0 _ yS)
  /-
    case intro.intro
    s : Set Real
    L : Real
    hL : Membership.mem s L
    U : Real
    hU : Membership.mem (upperBounds s) U
    this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
    f : Nat → Int
    hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
    hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
    hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
    hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
    ⊢ Exists fun x => IsLUB s x
  -/
  let g : CauSeq ℚ abs := ⟨fun n => f n / n, hg⟩
  /-
    case intro.intro
    s : Set Real
    L : Real
    hL : Membership.mem s L
    U : Real
    hU : Membership.mem (upperBounds s) U
    this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
    f : Nat → Int
    hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
    hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
    hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
    hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
    g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
    ⊢ Exists fun x => IsLUB s x
  -/
  refine ⟨mk g, ⟨fun x xS => ?_, fun y h => ?_⟩⟩
    /-
      case intro.intro.refine_1
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      ⊢ LE.le x (Real.mk g)
    -/
  · refine le_of_forall_ge_of_dense fun z xz => ?_
    /-
      case intro.intro.refine_1
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      xz : LT.lt z x
      ⊢ LE.le z (Real.mk g)
    -/
    cases' exists_nat_gt (x - z)⁻¹ with K hK
    /-
      case intro.intro.refine_1.intro
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      xz : LT.lt z x
      K : Nat
      hK : LT.lt (Inv.inv (HSub.hSub x z)) ↑K
      ⊢ LE.le z (Real.mk g)
    -/
    refine le_mk_of_forall_le ⟨K, fun n nK => ?_⟩
    /-
      case intro.intro.refine_1.intro
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      xz : LT.lt z x
      K : Nat
      hK : LT.lt (Inv.inv (HSub.hSub x z)) ↑K
      n : Nat
      nK : GE.ge n K
      ⊢ LE.le z ↑(↑g n)
    -/
    replace xz := sub_pos.2 xz
    /-
      case intro.intro.refine_1.intro
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      K : Nat
      hK : LT.lt (Inv.inv (HSub.hSub x z)) ↑K
      n : Nat
      nK : GE.ge n K
      xz : LT.lt 0 (HSub.hSub x z)
      ⊢ LE.le z ↑(↑g n)
    -/
    replace hK := hK.le.trans (Nat.cast_le.2 nK)
    /-
      case intro.intro.refine_1.intro
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      K n : Nat
      nK : GE.ge n K
      xz : LT.lt 0 (HSub.hSub x z)
      hK : LE.le (Inv.inv (HSub.hSub x z)) ↑n
      ⊢ LE.le z ↑(↑g n)
    -/
    have n0 : 0 < n := Nat.cast_pos.1 ((inv_pos.2 xz).trans_le hK)
    /-
      case intro.intro.refine_1.intro
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      K n : Nat
      nK : GE.ge n K
      xz : LT.lt 0 (HSub.hSub x z)
      hK : LE.le (Inv.inv (HSub.hSub x z)) ↑n
      n0 : LT.lt 0 n
      ⊢ LE.le z ↑(↑g n)
    -/
    refine le_trans ?_ (hf₂ _ n0 _ xS).le
    /-
      case intro.intro.refine_1.intro
      s : Set Real
      L : Real
      hL : Membership.mem s L
      U : Real
      hU : Membership.mem (upperBounds s) U
      this : ∀ (d : Nat), BddAbove (setOf fun m => Exists fun y => And (Membership.m …
      f : Nat → Int
      hf : ∀ (d : Nat), And (Membership.mem (setOf fun m => Exists fun y => And (Mem …
      hf₁ : ∀ (n : Nat), GT.gt n 0 → Exists fun y => And (Membership.mem s y) (LE.le …
      hf₂ : ∀ (n : Nat), GT.gt n 0 → ∀ (y : Real), Membership.mem s y → LT.lt (HSub. …
      hg : IsCauSeq abs fun n => HDiv.hDiv ↑(f n) ↑n
      g : CauSeq Rat abs := ⟨fun n => HDiv.hDiv ↑(f n) ↑n, hg⟩
      x : Real
      xS : Membership.mem s x
      z : Real
      K n : Nat
      nK : GE.ge n K
      xz : LT.lt 0 (HSub.hSub x z)
      hK : LE.le (Inv.inv (HSub.hSub x z)) ↑n
      n0 : LT.lt 0 n
      ⊢ LE.le z (HSub.hSub x (Inv.inv ↑n))
    -/
    rwa [le_sub_comm, inv_le_comm₀ (Nat.cast_pos.2 n0 : (_ : ℝ) < _) xz]
    /-
      🎉 no goals
    -/
  · exact
      mk_le_of_forall_le
        ⟨1, fun n n1 =>
          let ⟨x, xS, hx⟩ := hf₁ _ n1
          le_trans hx (h xS)⟩


/-- A nonempty, bounded below set of real numbers has a greatest lower bound. -/
theorem exists_isGLB (hne : s.Nonempty) (hbdd : BddBelow s) : ∃ x, IsGLB s x := by
  /-
    s : Set Real
    hne : s.Nonempty
    hbdd : BddBelow s
    ⊢ Exists fun x => IsGLB s x
  -/
  have hne' : (-s).Nonempty := Set.nonempty_neg.mpr hne
  /-
    s : Set Real
    hne : s.Nonempty
    hbdd : BddBelow s
    hne' : (Neg.neg s).Nonempty
    ⊢ Exists fun x => IsGLB s x
  -/
  have hbdd' : BddAbove (-s) := bddAbove_neg.mpr hbdd
  /-
    s : Set Real
    hne : s.Nonempty
    hbdd : BddBelow s
    hne' : (Neg.neg s).Nonempty
    hbdd' : BddAbove (Neg.neg s)
    ⊢ Exists fun x => IsGLB s x
  -/
  use -Classical.choose (Real.exists_isLUB hne' hbdd')
  /-
    case h
    s : Set Real
    hne : s.Nonempty
    hbdd : BddBelow s
    hne' : (Neg.neg s).Nonempty
    hbdd' : BddAbove (Neg.neg s)
    ⊢ IsGLB s (Neg.neg (Classical.choose ⋯))
  -/
  rw [← isLUB_neg]
  /-
    case h
    s : Set Real
    hne : s.Nonempty
    hbdd : BddBelow s
    hne' : (Neg.neg s).Nonempty
    hbdd' : BddAbove (Neg.neg s)
    ⊢ IsLUB (Neg.neg s) (Classical.choose ⋯)
  -/
  exact Classical.choose_spec (Real.exists_isLUB hne' hbdd')
  /-
    🎉 no goals
  -/


open scoped Classical in
noncomputable instance : SupSet ℝ :=
  ⟨fun s => if h : s.Nonempty ∧ BddAbove s then Classical.choose (exists_isLUB h.1 h.2) else 0⟩


open scoped Classical in
theorem sSup_def (s : Set ℝ) :
    sSup s = if h : s.Nonempty ∧ BddAbove s then Classical.choose (exists_isLUB h.1 h.2) else 0 :=
  rfl


protected theorem isLUB_sSup (h₁ : s.Nonempty) (h₂ : BddAbove s) : IsLUB s (sSup s) := by
  /-
    s : Set Real
    h₁ : s.Nonempty
    h₂ : BddAbove s
    ⊢ IsLUB s (SupSet.sSup s)
  -/
  simp only [sSup_def, dif_pos (And.intro h₁ h₂)]
  /-
    s : Set Real
    h₁ : s.Nonempty
    h₂ : BddAbove s
    ⊢ IsLUB s (Classical.choose ⋯)
  -/
  apply Classical.choose_spec
  /-
    🎉 no goals
  -/


noncomputable instance : InfSet ℝ :=
  ⟨fun s => -sSup (-s)⟩


theorem sInf_def (s : Set ℝ) : sInf s = -sSup (-s) := rfl


protected theorem isGLB_sInf (h₁ : s.Nonempty) (h₂ : BddBelow s) : IsGLB s (sInf s) := by
  /-
    s : Set Real
    h₁ : s.Nonempty
    h₂ : BddBelow s
    ⊢ IsGLB s (InfSet.sInf s)
  -/
  rw [sInf_def, ← isLUB_neg', neg_neg]
  /-
    s : Set Real
    h₁ : s.Nonempty
    h₂ : BddBelow s
    ⊢ IsLUB (Neg.neg s) (SupSet.sSup (Neg.neg s))
  -/
  exact Real.isLUB_sSup h₁.neg h₂.neg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-02")] alias is_glb_sInf := isGLB_sInf


noncomputable instance : ConditionallyCompleteLinearOrder ℝ where
  __ := Real.linearOrder
  __ := Real.lattice
  le_csSup s a hs ha := (Real.isLUB_sSup ⟨a, ha⟩ hs).1 ha
  csSup_le s a hs ha := (Real.isLUB_sSup hs ⟨a, ha⟩).2 ha
  csInf_le s a hs ha := (Real.isGLB_sInf ⟨a, ha⟩ hs).1 ha
  le_csInf s a hs ha := (Real.isGLB_sInf hs ⟨a, ha⟩).2 ha
                                   /-
                                     ι : Sort u_1
                                     f : ι → Real
                                     s✝ : Set Real
                                     a : Real
                                     s : Set Real
                                     hs : Not (BddAbove s)
                                     ⊢ Eq (SupSet.sSup s) (SupSet.sSup EmptyCollection.emptyCollection)
                                   -/
  csSup_of_not_bddAbove s hs := by simp [hs, sSup_def]
                                   /-
                                     🎉 no goals
                                   -/
                                   /-
                                     ι : Sort u_1
                                     f : ι → Real
                                     s✝ : Set Real
                                     a : Real
                                     s : Set Real
                                     hs : Not (BddBelow s)
                                     ⊢ Eq (InfSet.sInf s) (InfSet.sInf EmptyCollection.emptyCollection)
                                   -/
  csInf_of_not_bddBelow s hs := by simp [hs, sInf_def, sSup_def]
                                   /-
                                     🎉 no goals
                                   -/


theorem lt_sInf_add_pos (h : s.Nonempty) {ε : ℝ} (hε : 0 < ε) : ∃ a ∈ s, a < sInf s + ε :=
  exists_lt_of_csInf_lt h <| lt_add_of_pos_right _ hε


theorem add_neg_lt_sSup (h : s.Nonempty) {ε : ℝ} (hε : ε < 0) : ∃ a ∈ s, sSup s + ε < a :=
  exists_lt_of_lt_csSup h <| add_lt_iff_neg_left.2 hε


theorem sInf_le_iff (h : BddBelow s) (h' : s.Nonempty) :
    sInf s ≤ a ↔ ∀ ε, 0 < ε → ∃ x ∈ s, x < a + ε := by
  /-
    s : Set Real
    a : Real
    h : BddBelow s
    h' : s.Nonempty
    ⊢ Iff (LE.le (InfSet.sInf s) a) (∀ (ε : Real), LT.lt 0 ε → Exists fun x => And …
  -/
  rw [le_iff_forall_pos_lt_add]
  /-
    s : Set Real
    a : Real
    h : BddBelow s
    h' : s.Nonempty
    ⊢ Iff (∀ (ε : Real), LT.lt 0 ε → LT.lt (InfSet.sInf s) (HAdd.hAdd a ε)) (∀ (ε  …
  -/
  constructor <;> intro H ε ε_pos
    /-
      case mp
      s : Set Real
      a : Real
      h : BddBelow s
      h' : s.Nonempty
      H : ∀ (ε : Real), LT.lt 0 ε → LT.lt (InfSet.sInf s) (HAdd.hAdd a ε)
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ Exists fun x => And (Membership.mem s x) (LT.lt x (HAdd.hAdd a ε))
    -/
  · exact exists_lt_of_csInf_lt h' (H ε ε_pos)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      s : Set Real
      a : Real
      h : BddBelow s
      h' : s.Nonempty
      H : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (Membership.mem s x) (LT.lt  …
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ LT.lt (InfSet.sInf s) (HAdd.hAdd a ε)
    -/
  · rcases H ε ε_pos with ⟨x, x_in, hx⟩
    /-
      case mpr.intro.intro
      s : Set Real
      a : Real
      h : BddBelow s
      h' : s.Nonempty
      H : ∀ (ε : Real), LT.lt 0 ε → Exists fun x => And (Membership.mem s x) (LT.lt  …
      ε : Real
      ε_pos : LT.lt 0 ε
      x : Real
      x_in : Membership.mem s x
      hx : LT.lt x (HAdd.hAdd a ε)
      ⊢ LT.lt (InfSet.sInf s) (HAdd.hAdd a ε)
    -/
    exact csInf_lt_of_lt h x_in hx
    /-
      🎉 no goals
    -/


theorem le_sSup_iff (h : BddAbove s) (h' : s.Nonempty) :
    a ≤ sSup s ↔ ∀ ε, ε < 0 → ∃ x ∈ s, a + ε < x := by
  /-
    s : Set Real
    a : Real
    h : BddAbove s
    h' : s.Nonempty
    ⊢ Iff (LE.le a (SupSet.sSup s)) (∀ (ε : Real), LT.lt ε 0 → Exists fun x => And …
  -/
  rw [le_iff_forall_pos_lt_add]
  /-
    s : Set Real
    a : Real
    h : BddAbove s
    h' : s.Nonempty
    ⊢ Iff (∀ (ε : Real), LT.lt 0 ε → LT.lt a (HAdd.hAdd (SupSet.sSup s) ε)) (∀ (ε  …
  -/
  refine ⟨fun H ε ε_neg => ?_, fun H ε ε_pos => ?_⟩
    /-
      case refine_1
      s : Set Real
      a : Real
      h : BddAbove s
      h' : s.Nonempty
      H : ∀ (ε : Real), LT.lt 0 ε → LT.lt a (HAdd.hAdd (SupSet.sSup s) ε)
      ε : Real
      ε_neg : LT.lt ε 0
      ⊢ Exists fun x => And (Membership.mem s x) (LT.lt (HAdd.hAdd a ε) x)
    -/
  · exact exists_lt_of_lt_csSup h' (lt_sub_iff_add_lt.mp (H _ (neg_pos.mpr ε_neg)))
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      s : Set Real
      a : Real
      h : BddAbove s
      h' : s.Nonempty
      H : ∀ (ε : Real), LT.lt ε 0 → Exists fun x => And (Membership.mem s x) (LT.lt  …
      ε : Real
      ε_pos : LT.lt 0 ε
      ⊢ LT.lt a (HAdd.hAdd (SupSet.sSup s) ε)
    -/
  · rcases H _ (neg_lt_zero.mpr ε_pos) with ⟨x, x_in, hx⟩
    /-
      case refine_2.intro.intro
      s : Set Real
      a : Real
      h : BddAbove s
      h' : s.Nonempty
      H : ∀ (ε : Real), LT.lt ε 0 → Exists fun x => And (Membership.mem s x) (LT.lt  …
      ε : Real
      ε_pos : LT.lt 0 ε
      x : Real
      x_in : Membership.mem s x
      hx : LT.lt (HAdd.hAdd a (Neg.neg ε)) x
      ⊢ LT.lt a (HAdd.hAdd (SupSet.sSup s) ε)
    -/
    exact sub_lt_iff_lt_add.mp (lt_csSup_of_lt h x_in hx)
    /-
      🎉 no goals
    -/


@[simp]
theorem sSup_empty : sSup (∅ : Set ℝ) = 0 :=
                /-
                  ⊢ Not (And EmptyCollection.emptyCollection.Nonempty (BddAbove EmptyCollection. …
                -/
  dif_neg <| by simp
                /-
                  🎉 no goals
                -/


@[simp] lemma iSup_of_isEmpty [IsEmpty ι] (f : ι → ℝ) : ⨆ i, f i = 0 := by
  /-
    ι : Sort u_1
    inst✝ : IsEmpty ι
    f : ι → Real
    ⊢ Eq (iSup fun i => f i) 0
  -/
  dsimp [iSup]
  /-
    ι : Sort u_1
    inst✝ : IsEmpty ι
    f : ι → Real
    ⊢ Eq (SupSet.sSup (Set.range fun i => f i)) 0
  -/
  convert Real.sSup_empty
  /-
    case h.e'_2.h.e'_3
    ι : Sort u_1
    inst✝ : IsEmpty ι
    f : ι → Real
    ⊢ Eq (Set.range fun i => f i) EmptyCollection.emptyCollection
  -/
  rw [Set.range_eq_empty_iff]
  /-
    case h.e'_2.h.e'_3
    ι : Sort u_1
    inst✝ : IsEmpty ι
    f : ι → Real
    ⊢ IsEmpty ι
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[simp]
theorem iSup_const_zero : ⨆ _ : ι, (0 : ℝ) = 0 := by
  /-
    ι : Sort u_1
    ⊢ Eq (iSup fun x => 0) 0
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      h✝ : IsEmpty ι
      ⊢ Eq (iSup fun x => 0) 0
    -/
  · exact Real.iSup_of_isEmpty _
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      h✝ : Nonempty ι
      ⊢ Eq (iSup fun x => 0) 0
    -/
  · exact ciSup_const
    /-
      🎉 no goals
    -/


lemma sSup_of_not_bddAbove (hs : ¬BddAbove s) : sSup s = 0 := dif_neg fun h => hs h.2

lemma iSup_of_not_bddAbove (hf : ¬BddAbove (Set.range f)) : ⨆ i, f i = 0 := sSup_of_not_bddAbove hf


theorem sSup_univ : sSup (@Set.univ ℝ) = 0 := Real.sSup_of_not_bddAbove not_bddAbove_univ


@[simp]
                                                /-
                                                  ⊢ Eq (InfSet.sInf EmptyCollection.emptyCollection) 0
                                                -/
theorem sInf_empty : sInf (∅ : Set ℝ) = 0 := by simp [sInf_def, sSup_empty]
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] nonrec lemma iInf_of_isEmpty [IsEmpty ι] (f : ι → ℝ) : ⨅ i, f i = 0 := by
  /-
    ι : Sort u_1
    inst✝ : IsEmpty ι
    f : ι → Real
    ⊢ Eq (iInf fun i => f i) 0
  -/
  rw [iInf_of_isEmpty, sInf_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem iInf_const_zero : ⨅ _ : ι, (0 : ℝ) = 0 := by
  /-
    ι : Sort u_1
    ⊢ Eq (iInf fun x => 0) 0
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      h✝ : IsEmpty ι
      ⊢ Eq (iInf fun x => 0) 0
    -/
  · exact Real.iInf_of_isEmpty _
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      h✝ : Nonempty ι
      ⊢ Eq (iInf fun x => 0) 0
    -/
  · exact ciInf_const
    /-
      🎉 no goals
    -/


theorem sInf_of_not_bddBelow (hs : ¬BddBelow s) : sInf s = 0 :=
  neg_eq_zero.2 <| sSup_of_not_bddAbove <| mt bddAbove_neg.1 hs


theorem iInf_of_not_bddBelow (hf : ¬BddBelow (Set.range f)) : ⨅ i, f i = 0 :=
  sInf_of_not_bddBelow hf


/-- As `sSup s = 0` when `s` is an empty set of reals, it suffices to show that all elements of `s`
are at most some nonnegative number `a` to show that `sSup s ≤ a`.

See also `csSup_le`. -/
protected lemma sSup_le (hs : ∀ x ∈ s, x ≤ a) (ha : 0 ≤ a) : sSup s ≤ a := by
  /-
    s : Set Real
    a : Real
    hs : ∀ (x : Real), Membership.mem s x → LE.le x a
    ha : LE.le 0 a
    ⊢ LE.le (SupSet.sSup s) a
  -/
  obtain rfl | hs' := s.eq_empty_or_nonempty
  /-
    case inl
    a : Real
    ha : LE.le 0 a
    hs : ∀ (x : Real), Membership.mem EmptyCollection.emptyCollection x → LE.le x a
    ⊢ LE.le (SupSet.sSup EmptyCollection.emptyCollection) a
  -/
  exacts [sSup_empty.trans_le ha, csSup_le hs' hs]
  /-
    🎉 no goals
  -/


/-- As `⨆ i, f i = 0` when the domain of the real-valued function `f` is empty, it suffices to show
that all values of `f` are at most some nonnegative number `a` to show that `⨆ i, f i ≤ a`.

See also `ciSup_le`. -/
protected lemma iSup_le (hf : ∀ i, f i ≤ a) (ha : 0 ≤ a) : ⨆ i, f i ≤ a :=
  Real.sSup_le (Set.forall_mem_range.2 hf) ha


/-- As `sInf s = 0` when `s` is an empty set of reals, it suffices to show that all elements of `s`
are at least some nonpositive number `a` to show that `a ≤ sInf s`.

See also `le_csInf`. -/
protected lemma le_sInf (hs : ∀ x ∈ s, a ≤ x) (ha : a ≤ 0) : a ≤ sInf s := by
  /-
    s : Set Real
    a : Real
    hs : ∀ (x : Real), Membership.mem s x → LE.le a x
    ha : LE.le a 0
    ⊢ LE.le a (InfSet.sInf s)
  -/
  obtain rfl | hs' := s.eq_empty_or_nonempty
  /-
    case inl
    a : Real
    ha : LE.le a 0
    hs : ∀ (x : Real), Membership.mem EmptyCollection.emptyCollection x → LE.le a x
    ⊢ LE.le a (InfSet.sInf EmptyCollection.emptyCollection)
  -/
  exacts [ha.trans_eq sInf_empty.symm, le_csInf hs' hs]
  /-
    🎉 no goals
  -/


/-- As `⨅ i, f i = 0` when the domain of the real-valued function `f` is empty, it suffices to show
that all values of `f` are at least some nonpositive number `a` to show that `a ≤ ⨅ i, f i`.

See also `le_ciInf`. -/
protected lemma le_iInf (hf : ∀ i, a ≤ f i) (ha : a ≤ 0) : a ≤ ⨅ i, f i :=
  Real.le_sInf (Set.forall_mem_range.2 hf) ha


/-- As `sSup s = 0` when `s` is an empty set of reals, it suffices to show that all elements of `s`
are nonpositive to show that `sSup s ≤ 0`. -/
lemma sSup_nonpos (hs : ∀ x ∈ s, x ≤ 0) : sSup s ≤ 0 := Real.sSup_le hs le_rfl


/-- As `⨆ i, f i = 0` when the domain of the real-valued function `f` is empty,
it suffices to show that all values of `f` are nonpositive to show that `⨆ i, f i ≤ 0`. -/
lemma iSup_nonpos (hf : ∀ i, f i ≤ 0) : ⨆ i, f i ≤ 0 := Real.iSup_le hf le_rfl


/-- As `sInf s = 0` when `s` is an empty set of reals, it suffices to show that all elements of `s`
are nonnegative to show that `0 ≤ sInf s`. -/
lemma sInf_nonneg (hs : ∀ x ∈ s, 0 ≤ x) : 0 ≤ sInf s := Real.le_sInf hs le_rfl


/-- As `⨅ i, f i = 0` when the domain of the real-valued function `f` is empty,
it suffices to show that all values of `f` are nonnegative to show that `0 ≤ ⨅ i, f i`. -/
lemma iInf_nonneg (hf : ∀ i, 0 ≤ f i) : 0 ≤ iInf f := Real.le_iInf hf le_rfl


/-- As `sSup s = 0` when `s` is a set of reals that's unbounded above, it suffices to show that `s`
contains a nonnegative element to show that `0 ≤ sSup s`. -/
lemma sSup_nonneg' (hs : ∃ x ∈ s, 0 ≤ x) : 0 ≤ sSup s := by
  classical
  obtain ⟨x, hxs, hx⟩ := hs
  exact dite _ (fun h ↦ le_csSup_of_le h hxs hx) fun h ↦ (sSup_of_not_bddAbove h).ge


/-- As `⨆ i, f i = 0` when the real-valued function `f` is unbounded above,
it suffices to show that `f` takes a nonnegative value to show that `0 ≤ ⨆ i, f i`. -/
lemma iSup_nonneg' (hf : ∃ i, 0 ≤ f i) : 0 ≤ ⨆ i, f i := sSup_nonneg' <| Set.exists_range_iff.2 hf


/-- As `sInf s = 0` when `s` is a set of reals that's unbounded below, it suffices to show that `s`
contains a nonpositive element to show that `sInf s ≤ 0`. -/
lemma sInf_nonpos' (hs : ∃ x ∈ s, x ≤ 0) : sInf s ≤ 0 := by
  classical
  obtain ⟨x, hxs, hx⟩ := hs
  exact dite _ (fun h ↦ csInf_le_of_le h hxs hx) fun h ↦ (sInf_of_not_bddBelow h).le


/-- As `⨅ i, f i = 0` when the real-valued function `f` is unbounded below,
it suffices to show that `f` takes a nonpositive value to show that `0 ≤ ⨅ i, f i`. -/
lemma iInf_nonpos' (hf : ∃ i, f i ≤ 0) : ⨅ i, f i ≤ 0 := sInf_nonpos' <| Set.exists_range_iff.2 hf


/-- As `sSup s = 0` when `s` is a set of reals that's either empty or unbounded above,
it suffices to show that all elements of `s` are nonnegative to show that `0 ≤ sSup s`. -/
lemma sSup_nonneg (hs : ∀ x ∈ s, 0 ≤ x) : 0 ≤ sSup s := by
  /-
    s : Set Real
    hs : ∀ (x : Real), Membership.mem s x → LE.le 0 x
    ⊢ LE.le 0 (SupSet.sSup s)
  -/
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      hs : ∀ (x : Real), Membership.mem EmptyCollection.emptyCollection x → LE.le 0 x
      ⊢ LE.le 0 (SupSet.sSup EmptyCollection.emptyCollection)
    -/
  · exact sSup_empty.ge
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      s : Set Real
      hs : ∀ (x : Real), Membership.mem s x → LE.le 0 x
      x : Real
      hx : Membership.mem s x
      ⊢ LE.le 0 (SupSet.sSup s)
    -/
  · exact sSup_nonneg' ⟨x, hx, hs _ hx⟩
    /-
      🎉 no goals
    -/


/-- As `⨆ i, f i = 0` when the domain of the real-valued function `f` is empty or unbounded above,
it suffices to show that all values of `f` are nonnegative to show that `0 ≤ ⨆ i, f i`. -/
lemma iSup_nonneg (hf : ∀ i, 0 ≤ f i) : 0 ≤ ⨆ i, f i := sSup_nonneg <| Set.forall_mem_range.2 hf


/-- As `sInf s = 0` when `s` is a set of reals that's either empty or unbounded below,
it suffices to show that all elements of `s` are nonpositive to show that `sInf s ≤ 0`. -/
lemma sInf_nonpos (hs : ∀ x ∈ s, x ≤ 0) : sInf s ≤ 0 := by
  /-
    s : Set Real
    hs : ∀ (x : Real), Membership.mem s x → LE.le x 0
    ⊢ LE.le (InfSet.sInf s) 0
  -/
  obtain rfl | ⟨x, hx⟩ := s.eq_empty_or_nonempty
    /-
      case inl
      hs : ∀ (x : Real), Membership.mem EmptyCollection.emptyCollection x → LE.le x 0
      ⊢ LE.le (InfSet.sInf EmptyCollection.emptyCollection) 0
    -/
  · exact sInf_empty.le
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      s : Set Real
      hs : ∀ (x : Real), Membership.mem s x → LE.le x 0
      x : Real
      hx : Membership.mem s x
      ⊢ LE.le (InfSet.sInf s) 0
    -/
  · exact sInf_nonpos' ⟨x, hx, hs _ hx⟩
    /-
      🎉 no goals
    -/


/-- As `⨅ i, f i = 0` when the domain of the real-valued function `f` is empty or unbounded below,
it suffices to show that all values of `f` are nonpositive to show that `0 ≤ ⨅ i, f i`. -/
lemma iInf_nonpos (hf : ∀ i, f i ≤ 0) : ⨅ i, f i ≤ 0 := sInf_nonpos <| Set.forall_mem_range.2 hf


theorem sInf_le_sSup (s : Set ℝ) (h₁ : BddBelow s) (h₂ : BddAbove s) : sInf s ≤ sSup s := by
  /-
    s : Set Real
    h₁ : BddBelow s
    h₂ : BddAbove s
    ⊢ LE.le (InfSet.sInf s) (SupSet.sSup s)
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hne)
    /-
      case inl
      h₁ : BddBelow EmptyCollection.emptyCollection
      h₂ : BddAbove EmptyCollection.emptyCollection
      ⊢ LE.le (InfSet.sInf EmptyCollection.emptyCollection) (SupSet.sSup EmptyCollec …
    -/
  · rw [sInf_empty, sSup_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      s : Set Real
      h₁ : BddBelow s
      h₂ : BddAbove s
      hne : s.Nonempty
      ⊢ LE.le (InfSet.sInf s) (SupSet.sSup s)
    -/
  · exact csInf_le_csSup h₁ h₂ hne
    /-
      🎉 no goals
    -/


theorem cauSeq_converges (f : CauSeq ℝ abs) : ∃ x, f ≈ const abs x := by
  /-
    f : CauSeq Real abs
    ⊢ Exists fun x => HasEquiv.Equiv f (CauSeq.const abs x)
  -/
  let s := {x : ℝ | const abs x < f}
  /-
    f : CauSeq Real abs
    s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
    ⊢ Exists fun x => HasEquiv.Equiv f (CauSeq.const abs x)
  -/
  have lb : ∃ x, x ∈ s := exists_lt f
  have ub' : ∀ x, f < const abs x → ∀ y ∈ s, y ≤ x := fun x h y yS =>
    le_of_lt <| const_lt.1 <| CauSeq.lt_trans yS h
  /-
    f : CauSeq Real abs
    s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
    lb : Exists fun x => Membership.mem s x
    ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
    ⊢ Exists fun x => HasEquiv.Equiv f (CauSeq.const abs x)
  -/
  have ub : ∃ x, ∀ y ∈ s, y ≤ x := (exists_gt f).imp ub'
  /-
    f : CauSeq Real abs
    s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
    lb : Exists fun x => Membership.mem s x
    ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
    ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
    ⊢ Exists fun x => HasEquiv.Equiv f (CauSeq.const abs x)
  -/
  refine ⟨sSup s, ((lt_total _ _).resolve_left fun h => ?_).resolve_right fun h => ?_⟩
    /-
      case refine_1
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      h : LT.lt f (CauSeq.const abs (SupSet.sSup s))
      ⊢ False
    -/
  · rcases h with ⟨ε, ε0, i, ih⟩
    /-
      case refine_1.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub (CauSeq.const abs (SupSet.s …
      ⊢ False
    -/
    refine (csSup_le lb (ub' _ ?_)).not_lt (sub_lt_self _ (half_pos ε0))
    /-
      case refine_1.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub (CauSeq.const abs (SupSet.s …
      ⊢ LT.lt f (CauSeq.const abs (HSub.hSub (SupSet.sSup s) (HDiv.hDiv ε 2)))
    -/
    refine ⟨_, half_pos ε0, i, fun j ij => ?_⟩
    /-
      case refine_1.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub (CauSeq.const abs (SupSet.s …
      j : Nat
      ij : GE.ge j i
      ⊢ LE.le (HDiv.hDiv ε 2) (↑(HSub.hSub (CauSeq.const abs (HSub.hSub (SupSet.sSup …
    -/
    rw [sub_apply, const_apply, sub_right_comm, le_sub_iff_add_le, add_halves]
    /-
      case refine_1.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub (CauSeq.const abs (SupSet.s …
      j : Nat
      ij : GE.ge j i
      ⊢ LE.le ε (HSub.hSub (SupSet.sSup s) (↑f j))
    -/
    exact ih _ ij
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      h : LT.lt (CauSeq.const abs (SupSet.sSup s)) f
      ⊢ False
    -/
  · rcases h with ⟨ε, ε0, i, ih⟩
    /-
      case refine_2.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub f (CauSeq.const abs (SupSet …
      ⊢ False
    -/
    refine (le_csSup ub ?_).not_lt ((lt_add_iff_pos_left _).2 (half_pos ε0))
    /-
      case refine_2.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub f (CauSeq.const abs (SupSet …
      ⊢ Membership.mem s (HAdd.hAdd (HDiv.hDiv ε 2) (SupSet.sSup s))
    -/
    refine ⟨_, half_pos ε0, i, fun j ij => ?_⟩
    /-
      case refine_2.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub f (CauSeq.const abs (SupSet …
      j : Nat
      ij : GE.ge j i
      ⊢ LE.le (HDiv.hDiv ε 2) (↑(HSub.hSub f (CauSeq.const abs (HAdd.hAdd (HDiv.hDiv …
    -/
    rw [sub_apply, const_apply, add_comm, ← sub_sub, le_sub_iff_add_le, add_halves]
    /-
      case refine_2.intro.intro.intro
      f : CauSeq Real abs
      s : Set Real := setOf fun x => LT.lt (CauSeq.const abs x) f
      lb : Exists fun x => Membership.mem s x
      ub' : ∀ (x : Real), LT.lt f (CauSeq.const abs x) → ∀ (y : Real), Membership.me …
      ub : Exists fun x => ∀ (y : Real), Membership.mem s y → LE.le y x
      ε : Real
      ε0 : GT.gt ε 0
      i : Nat
      ih : ∀ (j : Nat), GE.ge j i → LE.le ε (↑(HSub.hSub f (CauSeq.const abs (SupSet …
      j : Nat
      ij : GE.ge j i
      ⊢ LE.le ε (HSub.hSub (↑f j) (SupSet.sSup s))
    -/
    exact ih _ ij
    /-
      🎉 no goals
    -/


instance : CauSeq.IsComplete ℝ abs :=
  ⟨cauSeq_converges⟩


theorem iInf_Ioi_eq_iInf_rat_gt {f : ℝ → ℝ} (x : ℝ) (hf : BddBelow (f '' Ioi x))
    (hf_mono : Monotone f) : ⨅ r : Ioi x, f r = ⨅ q : { q' : ℚ // x < q' }, f q := by
  /-
    f : Real → Real
    x : Real
    hf : BddBelow (Set.image f (Set.Ioi x))
    hf_mono : Monotone f
    ⊢ Eq (iInf fun r => f ↑r) (iInf fun q => f ↑↑q)
  -/
  refine le_antisymm ?_ ?_
  · have : Nonempty { r' : ℚ // x < ↑r' } := by
      obtain ⟨r, hrx⟩ := exists_rat_gt x
      exact ⟨⟨r, hrx⟩⟩
    /-
      case refine_1
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      this : Nonempty (Subtype fun r' => LT.lt x ↑r')
      ⊢ LE.le (iInf fun r => f ↑r) (iInf fun q => f ↑↑q)
    -/
    refine le_ciInf fun r => ?_
    /-
      case refine_1
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      this : Nonempty (Subtype fun r' => LT.lt x ↑r')
      r : Subtype fun q' => LT.lt x ↑q'
      ⊢ LE.le (iInf fun r => f ↑r) (f ↑↑r)
    -/
    obtain ⟨y, hxy, hyr⟩ := exists_rat_btwn r.prop
    /-
      case refine_1.intro.intro
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      this : Nonempty (Subtype fun r' => LT.lt x ↑r')
      r : Subtype fun q' => LT.lt x ↑q'
      y : Rat
      hxy : LT.lt x ↑y
      hyr : LT.lt ↑y ↑↑r
      ⊢ LE.le (iInf fun r => f ↑r) (f ↑↑r)
    -/
    refine ciInf_set_le hf (hxy.trans ?_)
    /-
      case refine_1.intro.intro
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      this : Nonempty (Subtype fun r' => LT.lt x ↑r')
      r : Subtype fun q' => LT.lt x ↑q'
      y : Rat
      hxy : LT.lt x ↑y
      hyr : LT.lt ↑y ↑↑r
      ⊢ LT.lt ↑y ↑↑r
    -/
    exact_mod_cast hyr
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      ⊢ LE.le (iInf fun q => f ↑↑q) (iInf fun r => f ↑r)
    -/
  · refine le_ciInf fun q => ?_
    /-
      case refine_2
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      q : ↑(Set.Ioi x)
      ⊢ LE.le (iInf fun q => f ↑↑q) (f ↑q)
    -/
    have hq := q.prop
    /-
      case refine_2
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      q : ↑(Set.Ioi x)
      hq : Membership.mem (Set.Ioi x) ↑q
      ⊢ LE.le (iInf fun q => f ↑↑q) (f ↑q)
    -/
    rw [mem_Ioi] at hq
    /-
      case refine_2
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      q : ↑(Set.Ioi x)
      hq : LT.lt x ↑q
      ⊢ LE.le (iInf fun q => f ↑↑q) (f ↑q)
    -/
    obtain ⟨y, hxy, hyq⟩ := exists_rat_btwn hq
    /-
      case refine_2.intro.intro
      f : Real → Real
      x : Real
      hf : BddBelow (Set.image f (Set.Ioi x))
      hf_mono : Monotone f
      q : ↑(Set.Ioi x)
      hq : LT.lt x ↑q
      y : Rat
      hxy : LT.lt x ↑y
      hyq : LT.lt ↑y ↑q
      ⊢ LE.le (iInf fun q => f ↑↑q) (f ↑q)
    -/
    refine (ciInf_le ?_ ?_).trans ?_
      /-
        case refine_2.intro.intro.refine_1
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        ⊢ BddBelow (Set.range fun q => f ↑↑q)
      -/
    · refine ⟨hf.some, fun z => ?_⟩
      /-
        case refine_2.intro.intro.refine_1
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        z : Real
        ⊢ Membership.mem (Set.range fun q => f ↑↑q) z → LE.le (Set.Nonempty.some hf) z
      -/
      rintro ⟨u, rfl⟩
      /-
        case refine_2.intro.intro.refine_1.intro
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        u : Subtype fun q' => LT.lt x ↑q'
        ⊢ LE.le (Set.Nonempty.some hf) ((fun q => f ↑↑q) u)
      -/
      suffices hfu : f u ∈ f '' Ioi x from hf.choose_spec hfu
      /-
        case refine_2.intro.intro.refine_1.intro
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        u : Subtype fun q' => LT.lt x ↑q'
        ⊢ Membership.mem (Set.image f (Set.Ioi x)) (f ↑↑u)
      -/
      exact ⟨u, u.prop, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.refine_2
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        ⊢ Subtype fun q' => LT.lt x ↑q'
      -/
    · exact ⟨y, hxy⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.intro.refine_3
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        ⊢ LE.le (f ↑↑⟨y, hxy⟩) (f ↑q)
      -/
    · refine hf_mono (le_trans ?_ hyq.le)
      /-
        case refine_2.intro.intro.refine_3
        f : Real → Real
        x : Real
        hf : BddBelow (Set.image f (Set.Ioi x))
        hf_mono : Monotone f
        q : ↑(Set.Ioi x)
        hq : LT.lt x ↑q
        y : Rat
        hxy : LT.lt x ↑y
        hyq : LT.lt ↑y ↑q
        ⊢ LE.le ↑↑⟨y, hxy⟩ ↑y
      -/
      norm_cast
      /-
        🎉 no goals
      -/


theorem not_bddAbove_coe : ¬ (BddAbove <| range (fun (x : ℚ) ↦ (x : ℝ))) := by
  /-
    ⊢ Not (BddAbove (Set.range fun x => ↑x))
  -/
  dsimp only [BddAbove, upperBounds]
  /-
    ⊢ Not (setOf fun x => ∀ ⦃a : Real⦄, Membership.mem (Set.range fun x => ↑x) a → …
  -/
  rw [Set.not_nonempty_iff_eq_empty]
  /-
    ⊢ Eq (setOf fun x => ∀ ⦃a : Real⦄, Membership.mem (Set.range fun x => ↑x) a →  …
  -/
  ext
  /-
    case h
    x✝ : Real
    ⊢ Iff (Membership.mem (setOf fun x => ∀ ⦃a : Real⦄, Membership.mem (Set.range  …
  -/
  simpa using exists_rat_gt _
  /-
    🎉 no goals
  -/


theorem not_bddBelow_coe : ¬ (BddBelow <| range (fun (x : ℚ) ↦ (x : ℝ))) := by
  /-
    ⊢ Not (BddBelow (Set.range fun x => ↑x))
  -/
  dsimp only [BddBelow, lowerBounds]
  /-
    ⊢ Not (setOf fun x => ∀ ⦃a : Real⦄, Membership.mem (Set.range fun x => ↑x) a → …
  -/
  rw [Set.not_nonempty_iff_eq_empty]
  /-
    ⊢ Eq (setOf fun x => ∀ ⦃a : Real⦄, Membership.mem (Set.range fun x => ↑x) a →  …
  -/
  ext
  /-
    case h
    x✝ : Real
    ⊢ Iff (Membership.mem (setOf fun x => ∀ ⦃a : Real⦄, Membership.mem (Set.range  …
  -/
  simpa using exists_rat_lt _
  /-
    🎉 no goals
  -/


theorem iUnion_Iic_rat : ⋃ r : ℚ, Iic (r : ℝ) = univ := by
  /-
    ⊢ Eq (Set.iUnion fun r => Set.Iic ↑r) Set.univ
  -/
  exact iUnion_Iic_of_not_bddAbove_range not_bddAbove_coe
  /-
    🎉 no goals
  -/


theorem iInter_Iic_rat : ⋂ r : ℚ, Iic (r : ℝ) = ∅ := by
  /-
    ⊢ Eq (Set.iInter fun r => Set.Iic ↑r) EmptyCollection.emptyCollection
  -/
  exact iInter_Iic_eq_empty_iff.mpr not_bddBelow_coe
  /-
    🎉 no goals
  -/


/-- Exponentiation is eventually larger than linear growth. -/
lemma exists_natCast_add_one_lt_pow_of_one_lt (ha : 1 < a) : ∃ m : ℕ, (m + 1 : ℝ) < a ^ m := by
  obtain ⟨k, posk, hk⟩ : ∃ k : ℕ, 0 < k ∧ 1 / k + 1 < a := by
    contrapose! ha
    refine le_of_forall_lt_rat_imp_le ?_
    intro q hq
    refine (ha q.den (by positivity)).trans ?_
    rw [← le_sub_iff_add_le, div_le_iff₀ (by positivity), sub_mul, one_mul]
    norm_cast at hq ⊢
    rw [← q.num_div_den, one_lt_div (by positivity)] at hq
    rw [q.mul_den_eq_num]
    norm_cast at hq ⊢
    omega
  /-
    case intro.intro
    a : Real
    ha : LT.lt 1 a
    k : Nat
    posk : LT.lt 0 k
    hk : LT.lt (HAdd.hAdd (HDiv.hDiv 1 ↑k) 1) a
    ⊢ Exists fun m => LT.lt (HAdd.hAdd (↑m) 1) (HPow.hPow a m)
  -/
  use 2 * k ^ 2
  calc
    ((2 * k ^ 2 : ℕ) + 1 : ℝ) ≤ 2 ^ (2 * k) := mod_cast Nat.two_mul_sq_add_one_le_two_pow_two_mul _
    _ = (1 / k * k + 1 : ℝ) ^ (2 * k) := by simp [posk.ne']; norm_num
    _ ≤ ((1 / k + 1) ^ k : ℝ) ^ (2 * k) := by gcongr; exact mul_add_one_le_add_one_pow (by simp) _
    _ = (1 / k + 1 : ℝ) ^ (2 * k ^ 2) := by rw [← pow_mul, mul_left_comm, sq]
    _ < a ^ (2 * k ^ 2) := by gcongr


