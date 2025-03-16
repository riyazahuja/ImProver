/-- Bundled monotone right-continuous real functions, used to construct Stieltjes measures. -/
structure StieltjesFunction where
  toFun : ℝ → ℝ
  mono' : Monotone toFun
  right_continuous' : ∀ x, ContinuousWithinAt toFun (Ici x) x


instance instCoeFun : CoeFun StieltjesFunction fun _ => ℝ → ℝ :=
  ⟨toFun⟩


@[ext] lemma ext {f g : StieltjesFunction} (h : ∀ x, f x = g x) : f = g := by
  /-
    f g : StieltjesFunction
    h : ∀ (x : Real), Eq (↑f x) (↑g x)
    ⊢ Eq f g
  -/
  exact (StieltjesFunction.mk.injEq ..).mpr (funext h)
  /-
    🎉 no goals
  -/


theorem mono : Monotone f :=
  f.mono'


theorem right_continuous (x : ℝ) : ContinuousWithinAt f (Ici x) x :=
  f.right_continuous' x


theorem rightLim_eq (f : StieltjesFunction) (x : ℝ) : Function.rightLim f x = f x := by
  /-
    f : StieltjesFunction
    x : Real
    ⊢ Eq (Function.rightLim (↑f) x) (↑f x)
  -/
  rw [← f.mono.continuousWithinAt_Ioi_iff_rightLim_eq, continuousWithinAt_Ioi_iff_Ici]
  /-
    f : StieltjesFunction
    x : Real
    ⊢ ContinuousWithinAt (↑f) (Set.Ici x) x
  -/
  exact f.right_continuous' x
  /-
    🎉 no goals
  -/


theorem iInf_Ioi_eq (f : StieltjesFunction) (x : ℝ) : ⨅ r : Ioi x, f r = f x := by
  /-
    f : StieltjesFunction
    x : Real
    ⊢ Eq (iInf fun r => ↑f ↑r) (↑f x)
  -/
  suffices Function.rightLim f x = ⨅ r : Ioi x, f r by rw [← this, f.rightLim_eq]
  /-
    f : StieltjesFunction
    x : Real
    ⊢ Eq (Function.rightLim (↑f) x) (iInf fun r => ↑f ↑r)
  -/
  rw [f.mono.rightLim_eq_sInf, sInf_image']
  /-
    f : StieltjesFunction
    x : Real
    ⊢ Ne (nhdsWithin x (Set.Ioi x)) Bot.bot
  -/
  rw [← neBot_iff]
  /-
    f : StieltjesFunction
    x : Real
    ⊢ (nhdsWithin x (Set.Ioi x)).NeBot
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem iInf_rat_gt_eq (f : StieltjesFunction) (x : ℝ) :
    ⨅ r : { r' : ℚ // x < r' }, f r = f x := by
  /-
    f : StieltjesFunction
    x : Real
    ⊢ Eq (iInf fun r => ↑f ↑↑r) (↑f x)
  -/
  rw [← iInf_Ioi_eq f x]
  /-
    f : StieltjesFunction
    x : Real
    ⊢ Eq (iInf fun r => ↑f ↑↑r) (iInf fun r => ↑f ↑r)
  -/
  refine (Real.iInf_Ioi_eq_iInf_rat_gt _ ?_ f.mono).symm
  /-
    f : StieltjesFunction
    x : Real
    ⊢ BddBelow (Set.image (↑f) (Set.Ioi x))
  -/
  refine ⟨f x, fun y => ?_⟩
  /-
    f : StieltjesFunction
    x y : Real
    ⊢ Membership.mem (Set.image (↑f) (Set.Ioi x)) y → LE.le (↑f x) y
  -/
  rintro ⟨y, hy_mem, rfl⟩
  /-
    case intro.intro
    f : StieltjesFunction
    x y : Real
    hy_mem : Membership.mem (Set.Ioi x) y
    ⊢ LE.le (↑f x) (↑f y)
  -/
  exact f.mono (le_of_lt hy_mem)
  /-
    🎉 no goals
  -/


/-- The identity of `ℝ` as a Stieltjes function, used to construct Lebesgue measure. -/
@[simps]
protected def id : StieltjesFunction where
  toFun := id
  mono' _ _ := id
  right_continuous' _ := continuousWithinAt_id


@[simp]
theorem id_leftLim (x : ℝ) : leftLim StieltjesFunction.id x = x :=
  tendsto_nhds_unique (StieltjesFunction.id.mono.tendsto_leftLim x) <|
    continuousAt_id.tendsto.mono_left nhdsWithin_le_nhds


instance instInhabited : Inhabited StieltjesFunction :=
  ⟨StieltjesFunction.id⟩


/-- Constant functions are Stieltjes function. -/
protected def const (c : ℝ) : StieltjesFunction where
  toFun := fun _ ↦ c
                  /-
                    f : StieltjesFunction
                    c x✝¹ x✝ : Real
                    ⊢ LE.le x✝¹ x✝ → LE.le ((fun x => c) x✝¹) ((fun x => c) x✝)
                  -/
  mono' _ _ := by simp
                  /-
                    🎉 no goals
                  -/
  right_continuous' _ := continuousWithinAt_const


@[simp] lemma const_apply (c x : ℝ) : (StieltjesFunction.const c) x = c := rfl


/-- The sum of two Stieltjes functions is a Stieltjes function. -/
protected def add (f g : StieltjesFunction) : StieltjesFunction where
  toFun := fun x => f x + g x
  mono' := f.mono.add g.mono
  right_continuous' := fun x => (f.right_continuous x).add (g.right_continuous x)


instance : AddZeroClass StieltjesFunction where
  add := StieltjesFunction.add
  zero := StieltjesFunction.const 0
  zero_add _ := ext fun _ ↦ zero_add _
  add_zero _ := ext fun _ ↦ add_zero _


instance : AddCommMonoid StieltjesFunction where
  nsmul n f := nsmulRec n f
  add_assoc _ _ _ := ext fun _ ↦ add_assoc _ _ _
  add_comm _ _ := ext fun _ ↦ add_comm _ _
  __ := StieltjesFunction.instAddZeroClass


instance : Module ℝ≥0 StieltjesFunction where
  smul c f := {
    toFun := fun x ↦ c * f x
    mono' := f.mono.const_mul c.2
    right_continuous' := fun x ↦ (f.right_continuous x).const_smul c.1}
  one_smul _ := ext fun _ ↦ one_mul _
  mul_smul _ _ _ := ext fun _ ↦ mul_assoc _ _ _
  smul_zero _ := ext fun _ ↦ mul_zero _
  smul_add _ _ _ := ext fun _ ↦ mul_add _ _ _
  add_smul _ _ _ := ext fun _ ↦ add_mul _ _ _
  zero_smul _ := ext fun _ ↦ zero_mul _


@[simp] lemma zero_apply (x : ℝ) : (0 : StieltjesFunction) x = 0 := rfl


@[simp] lemma add_apply (f g : StieltjesFunction) (x : ℝ) : (f + g) x = f x + g x := rfl


/-- If a function `f : ℝ → ℝ` is monotone, then the function mapping `x` to the right limit of `f`
at `x` is a Stieltjes function, i.e., it is monotone and right-continuous. -/
noncomputable def _root_.Monotone.stieltjesFunction {f : ℝ → ℝ} (hf : Monotone f) :
    StieltjesFunction where
  toFun := rightLim f
  mono' _ _ hxy := hf.rightLim hxy
  right_continuous' := by
    /-
      f✝ : StieltjesFunction
      f : Real → Real
      hf : Monotone f
      ⊢ ∀ (x : Real), ContinuousWithinAt (Function.rightLim f) (Set.Ici x) x
    -/
    intro x s hs
    obtain ⟨l, u, hlu, lus⟩ : ∃ l u : ℝ, rightLim f x ∈ Ioo l u ∧ Ioo l u ⊆ s :=
      mem_nhds_iff_exists_Ioo_subset.1 hs
    obtain ⟨y, xy, h'y⟩ : ∃ (y : ℝ), x < y ∧ Ioc x y ⊆ f ⁻¹' Ioo l u :=
      mem_nhdsGT_iff_exists_Ioc_subset.1 (hf.tendsto_rightLim x (Ioo_mem_nhds hlu.1 hlu.2))
    /-
      case intro.intro.intro.intro.intro
      f✝ : StieltjesFunction
      f : Real → Real
      hf : Monotone f
      x : Real
      s : Set Real
      hs : Membership.mem (nhds (Function.rightLim f x)) s
      l u : Real
      hlu : Membership.mem (Set.Ioo l u) (Function.rightLim f x)
      lus : HasSubset.Subset (Set.Ioo l u) s
      y : Real
      xy : LT.lt x y
      h'y : HasSubset.Subset (Set.Ioc x y) (Set.preimage f (Set.Ioo l u))
      ⊢ Membership.mem (Filter.map (Function.rightLim f) (nhdsWithin x (Set.Ici x))) s
    -/
    change ∀ᶠ y in 𝓝[≥] x, rightLim f y ∈ s
    /-
      case intro.intro.intro.intro.intro
      f✝ : StieltjesFunction
      f : Real → Real
      hf : Monotone f
      x : Real
      s : Set Real
      hs : Membership.mem (nhds (Function.rightLim f x)) s
      l u : Real
      hlu : Membership.mem (Set.Ioo l u) (Function.rightLim f x)
      lus : HasSubset.Subset (Set.Ioo l u) s
      y : Real
      xy : LT.lt x y
      h'y : HasSubset.Subset (Set.Ioc x y) (Set.preimage f (Set.Ioo l u))
      ⊢ Filter.Eventually (fun y => Membership.mem s (Function.rightLim f y)) (nhdsW …
    -/
    filter_upwards [Ico_mem_nhdsGE xy] with z hz
    /-
      case h
      f✝ : StieltjesFunction
      f : Real → Real
      hf : Monotone f
      x : Real
      s : Set Real
      hs : Membership.mem (nhds (Function.rightLim f x)) s
      l u : Real
      hlu : Membership.mem (Set.Ioo l u) (Function.rightLim f x)
      lus : HasSubset.Subset (Set.Ioo l u) s
      y : Real
      xy : LT.lt x y
      h'y : HasSubset.Subset (Set.Ioc x y) (Set.preimage f (Set.Ioo l u))
      z : Real
      hz : Membership.mem (Set.Ico x y) z
      ⊢ Membership.mem s (Function.rightLim f z)
    -/
    apply lus
    /-
      case h.a
      f✝ : StieltjesFunction
      f : Real → Real
      hf : Monotone f
      x : Real
      s : Set Real
      hs : Membership.mem (nhds (Function.rightLim f x)) s
      l u : Real
      hlu : Membership.mem (Set.Ioo l u) (Function.rightLim f x)
      lus : HasSubset.Subset (Set.Ioo l u) s
      y : Real
      xy : LT.lt x y
      h'y : HasSubset.Subset (Set.Ioc x y) (Set.preimage f (Set.Ioo l u))
      z : Real
      hz : Membership.mem (Set.Ico x y) z
      ⊢ Membership.mem (Set.Ioo l u) (Function.rightLim f z)
    -/
    refine ⟨hlu.1.trans_le (hf.rightLim hz.1), ?_⟩
    /-
      case h.a
      f✝ : StieltjesFunction
      f : Real → Real
      hf : Monotone f
      x : Real
      s : Set Real
      hs : Membership.mem (nhds (Function.rightLim f x)) s
      l u : Real
      hlu : Membership.mem (Set.Ioo l u) (Function.rightLim f x)
      lus : HasSubset.Subset (Set.Ioo l u) s
      y : Real
      xy : LT.lt x y
      h'y : HasSubset.Subset (Set.Ioc x y) (Set.preimage f (Set.Ioo l u))
      z : Real
      hz : Membership.mem (Set.Ico x y) z
      ⊢ LT.lt (Function.rightLim f z) u
    -/
    obtain ⟨a, za, ay⟩ : ∃ a : ℝ, z < a ∧ a < y := exists_between hz.2
    calc
      rightLim f z ≤ f a := hf.rightLim_le za
      _ < u := (h'y ⟨hz.1.trans_lt za, ay.le⟩).2


theorem _root_.Monotone.stieltjesFunction_eq {f : ℝ → ℝ} (hf : Monotone f) (x : ℝ) :
    hf.stieltjesFunction x = rightLim f x :=
  rfl


theorem countable_leftLim_ne (f : StieltjesFunction) : Set.Countable { x | leftLim f x ≠ f x } := by
  /-
    f : StieltjesFunction
    ⊢ (setOf fun x => Ne (Function.leftLim (↑f) x) (↑f x)).Countable
  -/
  refine Countable.mono ?_ f.mono.countable_not_continuousAt
  /-
    f : StieltjesFunction
    ⊢ HasSubset.Subset (setOf fun x => Ne (Function.leftLim (↑f) x) (↑f x)) (setOf …
  -/
  intro x hx h'x
  /-
    f : StieltjesFunction
    x : Real
    hx : Membership.mem (setOf fun x => Ne (Function.leftLim (↑f) x) (↑f x)) x
    h'x : ContinuousAt (↑f) x
    ⊢ False
  -/
  apply hx
  /-
    f : StieltjesFunction
    x : Real
    hx : Membership.mem (setOf fun x => Ne (Function.leftLim (↑f) x) (↑f x)) x
    h'x : ContinuousAt (↑f) x
    ⊢ Eq (Function.leftLim (↑f) x) (↑f x)
  -/
  exact tendsto_nhds_unique (f.mono.tendsto_leftLim x) (h'x.tendsto.mono_left nhdsWithin_le_nhds)
  /-
    🎉 no goals
  -/


/-- Length of an interval. This is the largest monotone function which correctly measures all
intervals. -/
def length (s : Set ℝ) : ℝ≥0∞ :=
  ⨅ (a) (b) (_ : s ⊆ Ioc a b), ofReal (f b - f a)


@[simp]
theorem length_empty : f.length ∅ = 0 :=
                                                                   /-
                                                                     f : StieltjesFunction
                                                                     ⊢ LE.le (iInf fun x => ENNReal.ofReal (HSub.hSub (↑f 0) (↑f 0))) 0
                                                                   -/
  nonpos_iff_eq_zero.1 <| iInf_le_of_le 0 <| iInf_le_of_le 0 <| by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem length_Ioc (a b : ℝ) : f.length (Ioc a b) = ofReal (f b - f a) := by
  refine
    le_antisymm (iInf_le_of_le a <| iInf₂_le b Subset.rfl)
      (le_iInf fun a' => le_iInf fun b' => le_iInf fun h => ENNReal.coe_le_coe.2 ?_)
  /-
    f : StieltjesFunction
    a b a' b' : Real
    h : HasSubset.Subset (Set.Ioc a b) (Set.Ioc a' b')
    ⊢ LE.le (HSub.hSub (↑f b) (↑f a)).toNNReal (HSub.hSub (↑f b') (↑f a')).toNNReal
  -/
  rcases le_or_lt b a with ab | ab
    /-
      case inl
      f : StieltjesFunction
      a b a' b' : Real
      h : HasSubset.Subset (Set.Ioc a b) (Set.Ioc a' b')
      ab : LE.le b a
      ⊢ LE.le (HSub.hSub (↑f b) (↑f a)).toNNReal (HSub.hSub (↑f b') (↑f a')).toNNReal
    -/
  · rw [Real.toNNReal_of_nonpos (sub_nonpos.2 (f.mono ab))]
    /-
      case inl
      f : StieltjesFunction
      a b a' b' : Real
      h : HasSubset.Subset (Set.Ioc a b) (Set.Ioc a' b')
      ab : LE.le b a
      ⊢ LE.le 0 (HSub.hSub (↑f b') (↑f a')).toNNReal
    -/
    apply zero_le
    /-
      🎉 no goals
    -/
  /-
    case inr
    f : StieltjesFunction
    a b a' b' : Real
    h : HasSubset.Subset (Set.Ioc a b) (Set.Ioc a' b')
    ab : LT.lt a b
    ⊢ LE.le (HSub.hSub (↑f b) (↑f a)).toNNReal (HSub.hSub (↑f b') (↑f a')).toNNReal
  -/
  cases' (Ioc_subset_Ioc_iff ab).1 h with h₁ h₂
  /-
    case inr.intro
    f : StieltjesFunction
    a b a' b' : Real
    h : HasSubset.Subset (Set.Ioc a b) (Set.Ioc a' b')
    ab : LT.lt a b
    h₁ : LE.le b b'
    h₂ : LE.le a' a
    ⊢ LE.le (HSub.hSub (↑f b) (↑f a)).toNNReal (HSub.hSub (↑f b') (↑f a')).toNNReal
  -/
  exact Real.toNNReal_le_toNNReal (sub_le_sub (f.mono h₁) (f.mono h₂))
  /-
    🎉 no goals
  -/


theorem length_mono {s₁ s₂ : Set ℝ} (h : s₁ ⊆ s₂) : f.length s₁ ≤ f.length s₂ :=
  iInf_mono fun _ => biInf_mono fun _ => h.trans


/-- The Stieltjes outer measure associated to a Stieltjes function. -/
protected def outer : OuterMeasure ℝ :=
  OuterMeasure.ofFunction f.length f.length_empty


theorem outer_le_length (s : Set ℝ) : f.outer s ≤ f.length s :=
  OuterMeasure.ofFunction_le _


/-- If a compact interval `[a, b]` is covered by a union of open interval `(c i, d i)`, then
`f b - f a ≤ ∑ f (d i) - f (c i)`. This is an auxiliary technical statement to prove the same
statement for half-open intervals, the point of the current statement being that one can use
compactness to reduce it to a finite sum, and argue by induction on the size of the covering set. -/
theorem length_subadditive_Icc_Ioo {a b : ℝ} {c d : ℕ → ℝ} (ss : Icc a b ⊆ ⋃ i, Ioo (c i) (d i)) :
    ofReal (f b - f a) ≤ ∑' i, ofReal (f (d i) - f (c i)) := by
  suffices
    ∀ (s : Finset ℕ) (b), Icc a b ⊆ (⋃ i ∈ (s : Set ℕ), Ioo (c i) (d i)) →
      (ofReal (f b - f a) : ℝ≥0∞) ≤ ∑ i ∈ s, ofReal (f (d i) - f (c i)) by
    rcases isCompact_Icc.elim_finite_subcover_image
        (fun (i : ℕ) (_ : i ∈ univ) => @isOpen_Ioo _ _ _ _ (c i) (d i)) (by simpa using ss) with
      ⟨s, _, hf, hs⟩
    have e : ⋃ i ∈ (hf.toFinset : Set ℕ), Ioo (c i) (d i) = ⋃ i ∈ s, Ioo (c i) (d i) := by
      simp only [Set.ext_iff, exists_prop, Finset.set_biUnion_coe, mem_iUnion, forall_const,
        Finite.mem_toFinset]
    rw [ENNReal.tsum_eq_iSup_sum]
    refine le_trans ?_ (le_iSup _ hf.toFinset)
    exact this hf.toFinset _ (by simpa only [e] )
  /-
    f : StieltjesFunction
    a b : Real
    c d : Nat → Real
    ss : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.Ioo (c i) (d i))
    ⊢ ∀ (s : Finset Nat) (b : Real), HasSubset.Subset (Set.Icc a b) (Set.iUnion fu …
  -/
  clear ss b
  /-
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    ⊢ ∀ (s : Finset Nat) (b : Real), HasSubset.Subset (Set.Icc a b) (Set.iUnion fu …
  -/
  refine fun s => Finset.strongInductionOn s fun s IH b cv => ?_
  /-
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.iUnion fun h => S …
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (s.sum fun i => ENNReal.ofR …
  -/
  rcases le_total b a with ab | ab
    /-
      case inl
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.iUnion fun h => S …
      ab : LE.le b a
      ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (s.sum fun i => ENNReal.ofR …
    -/
  · rw [ENNReal.ofReal_eq_zero.2 (sub_nonpos.2 (f.mono ab))]
    /-
      case inl
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.iUnion fun h => S …
      ab : LE.le b a
      ⊢ LE.le 0 (s.sum fun i => ENNReal.ofReal (HSub.hSub (↑f (d i)) (↑f (c i))))
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/
  /-
    case inr
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.iUnion fun h => S …
    ab : LE.le a b
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (s.sum fun i => ENNReal.ofR …
  -/
  have := cv ⟨ab, le_rfl⟩
  simp only [Finset.mem_coe, gt_iff_lt, not_lt, mem_iUnion, mem_Ioo, exists_and_left,
    exists_prop] at this
  /-
    case inr
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.iUnion fun h => S …
    ab : LE.le a b
    this : Exists fun i => And (LT.lt (c i) b) (And (Membership.mem s i) (LT.lt b  …
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (s.sum fun i => ENNReal.ofR …
  -/
  rcases this with ⟨i, cb, is, bd⟩
  /-
    case inr.intro.intro.intro
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i => Set.iUnion fun h => S …
    ab : LE.le a b
    i : Nat
    cb : LT.lt (c i) b
    is : Membership.mem s i
    bd : LT.lt b (d i)
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (s.sum fun i => ENNReal.ofR …
  -/
  rw [← Finset.insert_erase is] at cv ⊢
  /-
    case inr.intro.intro.intro
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    ab : LE.le a b
    i : Nat
    cv : HasSubset.Subset (Set.Icc a b) (Set.iUnion fun i_1 => Set.iUnion fun h => …
    cb : LT.lt (c i) b
    is : Membership.mem s i
    bd : LT.lt b (d i)
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) ((Insert.insert i (s.erase  …
  -/
  rw [Finset.coe_insert, biUnion_insert] at cv
  /-
    case inr.intro.intro.intro
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    ab : LE.le a b
    i : Nat
    cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
    cb : LT.lt (c i) b
    is : Membership.mem s i
    bd : LT.lt b (d i)
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) ((Insert.insert i (s.erase  …
  -/
  rw [Finset.sum_insert (Finset.not_mem_erase _ _)]
  /-
    case inr.intro.intro.intro
    f : StieltjesFunction
    a : Real
    c d : Nat → Real
    s✝ s : Finset Nat
    IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
    b : Real
    ab : LE.le a b
    i : Nat
    cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
    cb : LT.lt (c i) b
    is : Membership.mem s i
    bd : LT.lt b (d i)
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (HAdd.hAdd (ENNReal.ofReal  …
  -/
  refine le_trans ?_ (add_le_add_left (IH _ (Finset.erase_ssubset is) (c i) ?_) _)
    /-
      case inr.intro.intro.intro.refine_1
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      ab : LE.le a b
      i : Nat
      cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
      cb : LT.lt (c i) b
      is : Membership.mem s i
      bd : LT.lt b (d i)
      ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (HAdd.hAdd (ENNReal.ofReal  …
    -/
  · refine le_trans (ENNReal.ofReal_le_ofReal ?_) ENNReal.ofReal_add_le
    /-
      case inr.intro.intro.intro.refine_1
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      ab : LE.le a b
      i : Nat
      cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
      cb : LT.lt (c i) b
      is : Membership.mem s i
      bd : LT.lt b (d i)
      ⊢ LE.le (HSub.hSub (↑f b) (↑f a)) (HAdd.hAdd (HSub.hSub (↑f (d i)) (↑f (c i))) …
    -/
    rw [sub_add_sub_cancel]
    /-
      case inr.intro.intro.intro.refine_1
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      ab : LE.le a b
      i : Nat
      cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
      cb : LT.lt (c i) b
      is : Membership.mem s i
      bd : LT.lt b (d i)
      ⊢ LE.le (HSub.hSub (↑f b) (↑f a)) (HSub.hSub (↑f (d i)) (↑f a))
    -/
    exact sub_le_sub_right (f.mono bd.le) _
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro.refine_2
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      ab : LE.le a b
      i : Nat
      cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
      cb : LT.lt (c i) b
      is : Membership.mem s i
      bd : LT.lt b (d i)
      ⊢ HasSubset.Subset (Set.Icc a (c i)) (Set.iUnion fun i_1 => Set.iUnion fun h = …
    -/
  · rintro x ⟨h₁, h₂⟩
    /-
      case inr.intro.intro.intro.refine_2.intro
      f : StieltjesFunction
      a : Real
      c d : Nat → Real
      s✝ s : Finset Nat
      IH : ∀ (t : Finset Nat), HasSSubset.SSubset t s → ∀ (b : Real), HasSubset.Subs …
      b : Real
      ab : LE.le a b
      i : Nat
      cv : HasSubset.Subset (Set.Icc a b) (Union.union (Set.Ioo (c i) (d i)) (Set.iU …
      cb : LT.lt (c i) b
      is : Membership.mem s i
      bd : LT.lt b (d i)
      x : Real
      h₁ : LE.le a x
      h₂ : LE.le x (c i)
      ⊢ Membership.mem (Set.iUnion fun i_1 => Set.iUnion fun h => Set.Ioo (c i_1) (d …
    -/
    exact (cv ⟨h₁, le_trans h₂ (le_of_lt cb)⟩).resolve_left (mt And.left (not_lt_of_le h₂))
    /-
      🎉 no goals
    -/


@[simp]
theorem outer_Ioc (a b : ℝ) : f.outer (Ioc a b) = ofReal (f b - f a) := by
  /- It suffices to show that, if `(a, b]` is covered by sets `s i`, then `f b - f a` is bounded
    by `∑ f.length (s i) + ε`. The difficulty is that `f.length` is expressed in terms of half-open
    intervals, while we would like to have a compact interval covered by open intervals to use
    compactness and finite sums, as provided by `length_subadditive_Icc_Ioo`. The trick is to use
    the right-continuity of `f`. If `a'` is close enough to `a` on its right, then `[a', b]` is
    still covered by the sets `s i` and moreover `f b - f a'` is very close to `f b - f a`
    (up to `ε/2`).
    Also, by definition one can cover `s i` by a half-closed interval `(p i, q i]` with `f`-length
    very close to that of `s i` (within a suitably small `ε' i`, say). If one moves `q i` very
    slightly to the right, then the `f`-length will change very little by right continuity, and we
    will get an open interval `(p i, q' i)` covering `s i` with `f (q' i) - f (p i)` within `ε' i`
    of the `f`-length of `s i`. -/
  refine
    le_antisymm
      (by
        rw [← f.length_Ioc]
        apply outer_le_length)
      (le_iInf₂ fun s hs => ENNReal.le_of_forall_pos_le_add fun ε εpos h => ?_)
  /-
    f : StieltjesFunction
    a b : Real
    s : Nat → Set Real
    hs : HasSubset.Subset (Set.Ioc a b) (Set.iUnion fun i => s i)
    ε : NNReal
    εpos : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (s i)) Top.top
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (HAdd.hAdd (tsum fun i => f …
  -/
  let δ := ε / 2
  /-
    f : StieltjesFunction
    a b : Real
    s : Nat → Set Real
    hs : HasSubset.Subset (Set.Ioc a b) (Set.iUnion fun i => s i)
    ε : NNReal
    εpos : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (s i)) Top.top
    δ : NNReal := HDiv.hDiv ε 2
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (HAdd.hAdd (tsum fun i => f …
  -/
  have δpos : 0 < (δ : ℝ≥0∞) := by simpa [δ] using εpos.ne'
  /-
    f : StieltjesFunction
    a b : Real
    s : Nat → Set Real
    hs : HasSubset.Subset (Set.Ioc a b) (Set.iUnion fun i => s i)
    ε : NNReal
    εpos : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (s i)) Top.top
    δ : NNReal := HDiv.hDiv ε 2
    δpos : LT.lt 0 ↑δ
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (HAdd.hAdd (tsum fun i => f …
  -/
  rcases ENNReal.exists_pos_sum_of_countable δpos.ne' ℕ with ⟨ε', ε'0, hε⟩
  obtain ⟨a', ha', aa'⟩ : ∃ a', f a' - f a < δ ∧ a < a' := by
    have A : ContinuousWithinAt (fun r => f r - f a) (Ioi a) a := by
      refine ContinuousWithinAt.sub ?_ continuousWithinAt_const
      exact (f.right_continuous a).mono Ioi_subset_Ici_self
    have B : f a - f a < δ := by rwa [sub_self, NNReal.coe_pos, ← ENNReal.coe_pos]
    exact (((tendsto_order.1 A).2 _ B).and self_mem_nhdsWithin).exists
  have : ∀ i, ∃ p : ℝ × ℝ, s i ⊆ Ioo p.1 p.2 ∧
      (ofReal (f p.2 - f p.1) : ℝ≥0∞) < f.length (s i) + ε' i := by
    intro i
    have hl :=
      ENNReal.lt_add_right ((ENNReal.le_tsum i).trans_lt h).ne (ENNReal.coe_ne_zero.2 (ε'0 i).ne')
    conv at hl =>
      lhs
      rw [length]
    simp only [iInf_lt_iff, exists_prop] at hl
    rcases hl with ⟨p, q', spq, hq'⟩
    have : ContinuousWithinAt (fun r => ofReal (f r - f p)) (Ioi q') q' := by
      apply ENNReal.continuous_ofReal.continuousAt.comp_continuousWithinAt
      refine ContinuousWithinAt.sub ?_ continuousWithinAt_const
      exact (f.right_continuous q').mono Ioi_subset_Ici_self
    rcases (((tendsto_order.1 this).2 _ hq').and self_mem_nhdsWithin).exists with ⟨q, hq, q'q⟩
    exact ⟨⟨p, q⟩, spq.trans (Ioc_subset_Ioo_right q'q), hq⟩
  /-
    case intro.intro.intro.intro
    f : StieltjesFunction
    a b : Real
    s : Nat → Set Real
    hs : HasSubset.Subset (Set.Ioc a b) (Set.iUnion fun i => s i)
    ε : NNReal
    εpos : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (s i)) Top.top
    δ : NNReal := HDiv.hDiv ε 2
    δpos : LT.lt 0 ↑δ
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑δ
    a' : Real
    ha' : LT.lt (HSub.hSub (↑f a') (↑f a)) ↑δ
    aa' : LT.lt a a'
    this : ∀ (i : Nat), Exists fun p => And (HasSubset.Subset (s i) (Set.Ioo p.1 p …
    ⊢ LE.le (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a))) (HAdd.hAdd (tsum fun i => f …
  -/
  choose g hg using this
  have I_subset : Icc a' b ⊆ ⋃ i, Ioo (g i).1 (g i).2 :=
    calc
      Icc a' b ⊆ Ioc a b := fun x hx => ⟨aa'.trans_le hx.1, hx.2⟩
      _ ⊆ ⋃ i, s i := hs
      _ ⊆ ⋃ i, Ioo (g i).1 (g i).2 := iUnion_mono fun i => (hg i).1
  calc
    ofReal (f b - f a) = ofReal (f b - f a' + (f a' - f a)) := by rw [sub_add_sub_cancel]
    _ ≤ ofReal (f b - f a') + ofReal (f a' - f a) := ENNReal.ofReal_add_le
    _ ≤ ∑' i, ofReal (f (g i).2 - f (g i).1) + ofReal δ :=
      (add_le_add (f.length_subadditive_Icc_Ioo I_subset) (ENNReal.ofReal_le_ofReal ha'.le))
    _ ≤ ∑' i, (f.length (s i) + ε' i) + δ :=
      (add_le_add (ENNReal.tsum_le_tsum fun i => (hg i).2.le)
        (by simp only [ENNReal.ofReal_coe_nnreal, le_rfl]))
    _ = ∑' i, f.length (s i) + ∑' i, (ε' i : ℝ≥0∞) + δ := by rw [ENNReal.tsum_add]
    _ ≤ ∑' i, f.length (s i) + δ + δ := add_le_add (add_le_add le_rfl hε.le) le_rfl
    _ = ∑' i : ℕ, f.length (s i) + ε := by simp [δ, add_assoc, ENNReal.add_halves]


theorem measurableSet_Ioi {c : ℝ} : MeasurableSet[f.outer.caratheodory] (Ioi c) := by
  /-
    f : StieltjesFunction
    c : Real
    ⊢ MeasurableSet (Set.Ioi c)
  -/
  refine OuterMeasure.ofFunction_caratheodory fun t => ?_
  /-
    f : StieltjesFunction
    c : Real
    t : Set Real
    ⊢ LE.le (HAdd.hAdd (f.length (Inter.inter t (Set.Ioi c))) (f.length (SDiff.sdi …
  -/
  refine le_iInf fun a => le_iInf fun b => le_iInf fun h => ?_
  refine
    le_trans
      (add_le_add (f.length_mono <| inter_subset_inter_left _ h)
        (f.length_mono <| diff_subset_diff_left h)) ?_
  /-
    f : StieltjesFunction
    c : Real
    t : Set Real
    a b : Real
    h : HasSubset.Subset t (Set.Ioc a b)
    ⊢ LE.le (HAdd.hAdd (f.length (Inter.inter (Set.Ioc a b) (Set.Ioi c))) (f.lengt …
  -/
  rcases le_total a c with hac | hac <;> rcases le_total b c with hbc | hbc
  · simp only [Ioc_inter_Ioi, f.length_Ioc, hac, hbc, le_refl, Ioc_eq_empty,
      max_eq_right, min_eq_left, Ioc_diff_Ioi, f.length_empty, zero_add, not_lt]
  · simp only [hac, hbc, Ioc_inter_Ioi, Ioc_diff_Ioi, f.length_Ioc, min_eq_right,
      ← ENNReal.ofReal_add, f.mono hac, f.mono hbc, sub_nonneg,
      sub_add_sub_cancel, le_refl,
      max_eq_right]
  · simp only [hbc, le_refl, Ioc_eq_empty, Ioc_inter_Ioi, min_eq_left, Ioc_diff_Ioi, f.length_empty,
      zero_add, or_true, le_sup_iff, f.length_Ioc, not_lt]
  · simp only [hac, hbc, Ioc_inter_Ioi, Ioc_diff_Ioi, f.length_Ioc, min_eq_right,
      le_refl, Ioc_eq_empty, add_zero, max_eq_left, f.length_empty, not_lt]


theorem outer_trim : f.outer.trim = f.outer := by
  /-
    f : StieltjesFunction
    ⊢ Eq f.outer.trim f.outer
  -/
  refine le_antisymm (fun s => ?_) (OuterMeasure.le_trim _)
  /-
    f : StieltjesFunction
    s : Set Real
    ⊢ LE.le (f.outer.trim s) (f.outer s)
  -/
  rw [OuterMeasure.trim_eq_iInf]
  /-
    f : StieltjesFunction
    s : Set Real
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => f.outer t) (f.outer s)
  -/
  refine le_iInf fun t => le_iInf fun ht => ENNReal.le_of_forall_pos_le_add fun ε ε0 h => ?_
  /-
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => f.outer t) (HAdd.hAdd (tsum …
  -/
  rcases ENNReal.exists_pos_sum_of_countable (ENNReal.coe_pos.2 ε0).ne' ℕ with ⟨ε', ε'0, hε⟩
  /-
    case intro.intro
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => f.outer t) (HAdd.hAdd (tsum …
  -/
  refine le_trans ?_ (add_le_add_left (le_of_lt hε) _)
  /-
    case intro.intro
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => f.outer t) (HAdd.hAdd (tsum …
  -/
  rw [← ENNReal.tsum_add]
  choose g hg using
    show ∀ i, ∃ s, t i ⊆ s ∧ MeasurableSet s ∧ f.outer s ≤ f.length (t i) + ofReal (ε' i) by
      intro i
      have hl :=
        ENNReal.lt_add_right ((ENNReal.le_tsum i).trans_lt h).ne (ENNReal.coe_pos.2 (ε'0 i)).ne'
      conv at hl =>
        lhs
        rw [length]
      simp only [iInf_lt_iff] at hl
      rcases hl with ⟨a, b, h₁, h₂⟩
      rw [← f.outer_Ioc] at h₂
      exact ⟨_, h₁, measurableSet_Ioc, le_of_lt <| by simpa using h₂⟩
  /-
    case intro.intro
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    g : Nat → Set Real
    hg : ∀ (i : Nat), And (HasSubset.Subset (t i) (g i)) (And (MeasurableSet (g i) …
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => f.outer t) (tsum fun a => H …
  -/
  simp only [ofReal_coe_nnreal] at hg
  /-
    case intro.intro
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    g : Nat → Set Real
    hg : ∀ (i : Nat), And (HasSubset.Subset (t i) (g i)) (And (MeasurableSet (g i) …
    ⊢ LE.le (iInf fun t => iInf fun x => iInf fun x => f.outer t) (tsum fun a => H …
  -/
  apply iInf_le_of_le (iUnion g) _
  /-
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    g : Nat → Set Real
    hg : ∀ (i : Nat), And (HasSubset.Subset (t i) (g i)) (And (MeasurableSet (g i) …
    ⊢ LE.le (iInf fun x => iInf fun x => f.outer (Set.iUnion g)) (tsum fun a => HA …
  -/
  apply iInf_le_of_le (ht.trans <| iUnion_mono fun i => (hg i).1) _
  /-
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    g : Nat → Set Real
    hg : ∀ (i : Nat), And (HasSubset.Subset (t i) (g i)) (And (MeasurableSet (g i) …
    ⊢ LE.le (iInf fun x => f.outer (Set.iUnion g)) (tsum fun a => HAdd.hAdd (f.len …
  -/
  apply iInf_le_of_le (MeasurableSet.iUnion fun i => (hg i).2.1) _
  /-
    f : StieltjesFunction
    s : Set Real
    t : Nat → Set Real
    ht : HasSubset.Subset s (Set.iUnion fun i => t i)
    ε : NNReal
    ε0 : LT.lt 0 ε
    h : LT.lt (tsum fun i => f.length (t i)) Top.top
    ε' : Nat → NNReal
    ε'0 : ∀ (i : Nat), LT.lt 0 (ε' i)
    hε : LT.lt (tsum fun i => ↑(ε' i)) ↑ε
    g : Nat → Set Real
    hg : ∀ (i : Nat), And (HasSubset.Subset (t i) (g i)) (And (MeasurableSet (g i) …
    ⊢ LE.le (f.outer (Set.iUnion g)) (tsum fun a => HAdd.hAdd (f.length (t a)) ↑(ε …
  -/
  exact le_trans (measure_iUnion_le _) (ENNReal.tsum_le_tsum fun i => (hg i).2.2)
  /-
    🎉 no goals
  -/


theorem borel_le_measurable : borel ℝ ≤ f.outer.caratheodory := by
  /-
    f : StieltjesFunction
    ⊢ LE.le (borel Real) f.outer.caratheodory
  -/
  rw [borel_eq_generateFrom_Ioi]
  /-
    f : StieltjesFunction
    ⊢ LE.le (MeasurableSpace.generateFrom (Set.range Set.Ioi)) f.outer.caratheodory
  -/
  refine MeasurableSpace.generateFrom_le ?_
  /-
    f : StieltjesFunction
    ⊢ ∀ (t : Set Real), Membership.mem (Set.range Set.Ioi) t → MeasurableSet t
  -/
  simp +contextual [f.measurableSet_Ioi]
  /-
    🎉 no goals
  -/


/-- The measure associated to a Stieltjes function, giving mass `f b - f a` to the
interval `(a, b]`. -/
protected irreducible_def measure : Measure ℝ where
  toOuterMeasure := f.outer
  m_iUnion _s hs := f.outer.iUnion_eq_of_caratheodory fun i => f.borel_le_measurable _ (hs i)
  trim_le := f.outer_trim.le


@[simp]
theorem measure_Ioc (a b : ℝ) : f.measure (Ioc a b) = ofReal (f b - f a) := by
  /-
    f : StieltjesFunction
    a b : Real
    ⊢ Eq (f.measure (Set.Ioc a b)) (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a)))
  -/
  rw [StieltjesFunction.measure]
  /-
    f : StieltjesFunction
    a b : Real
    ⊢ Eq ({ toOuterMeasure := f.outer, m_iUnion := ⋯, trim_le := ⋯ } (Set.Ioc a b) …
  -/
  exact f.outer_Ioc a b
  /-
    🎉 no goals
  -/


@[simp]
theorem measure_singleton (a : ℝ) : f.measure {a} = ofReal (f a - leftLim f a) := by
  obtain ⟨u, u_mono, u_lt_a, u_lim⟩ :
    ∃ u : ℕ → ℝ, StrictMono u ∧ (∀ n : ℕ, u n < a) ∧ Tendsto u atTop (𝓝 a) :=
    exists_seq_strictMono_tendsto a
  have A : {a} = ⋂ n, Ioc (u n) a := by
    refine Subset.antisymm (fun x hx => by simp [mem_singleton_iff.1 hx, u_lt_a]) fun x hx => ?_
    simp? at hx says simp only [mem_iInter, mem_Ioc] at hx
    have : a ≤ x := le_of_tendsto' u_lim fun n => (hx n).1.le
    simp [le_antisymm this (hx 0).2]
  have L1 : Tendsto (fun n => f.measure (Ioc (u n) a)) atTop (𝓝 (f.measure {a})) := by
    rw [A]
    refine tendsto_measure_iInter_atTop (fun n => nullMeasurableSet_Ioc)
      (fun m n hmn => ?_) ?_
    · exact Ioc_subset_Ioc_left (u_mono.monotone hmn)
    · exact ⟨0, by simpa only [measure_Ioc] using ENNReal.ofReal_ne_top⟩
  have L2 :
      Tendsto (fun n => f.measure (Ioc (u n) a)) atTop (𝓝 (ofReal (f a - leftLim f a))) := by
    simp only [measure_Ioc]
    have : Tendsto (fun n => f (u n)) atTop (𝓝 (leftLim f a)) := by
      apply (f.mono.tendsto_leftLim a).comp
      exact
        tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _ u_lim
          (Eventually.of_forall fun n => u_lt_a n)
    exact ENNReal.continuous_ofReal.continuousAt.tendsto.comp (tendsto_const_nhds.sub this)
  /-
    case intro.intro.intro
    f : StieltjesFunction
    a : Real
    u : Nat → Real
    u_mono : StrictMono u
    u_lt_a : ∀ (n : Nat), LT.lt (u n) a
    u_lim : Filter.Tendsto u Filter.atTop (nhds a)
    A : Eq (Singleton.singleton a) (Set.iInter fun n => Set.Ioc (u n) a)
    L1 : Filter.Tendsto (fun n => f.measure (Set.Ioc (u n) a)) Filter.atTop (nhds  …
    L2 : Filter.Tendsto (fun n => f.measure (Set.Ioc (u n) a)) Filter.atTop (nhds  …
    ⊢ Eq (f.measure (Singleton.singleton a)) (ENNReal.ofReal (HSub.hSub (↑f a) (Fu …
  -/
  exact tendsto_nhds_unique L1 L2
  /-
    🎉 no goals
  -/


@[simp]
theorem measure_Icc (a b : ℝ) : f.measure (Icc a b) = ofReal (f b - leftLim f a) := by
  /-
    f : StieltjesFunction
    a b : Real
    ⊢ Eq (f.measure (Set.Icc a b)) (ENNReal.ofReal (HSub.hSub (↑f b) (Function.lef …
  -/
  rcases le_or_lt a b with (hab | hab)
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le a b
      ⊢ Eq (f.measure (Set.Icc a b)) (ENNReal.ofReal (HSub.hSub (↑f b) (Function.lef …
    -/
  · have A : Disjoint {a} (Ioc a b) := by simp
    simp [← Icc_union_Ioc_eq_Icc le_rfl hab, -singleton_union, ← ENNReal.ofReal_add,
      f.mono.leftLim_le, measure_union A measurableSet_Ioc, f.mono hab]
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt b a
      ⊢ Eq (f.measure (Set.Icc a b)) (ENNReal.ofReal (HSub.hSub (↑f b) (Function.lef …
    -/
  · simp only [hab, measure_empty, Icc_eq_empty, not_le]
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt b a
      ⊢ Eq 0 (ENNReal.ofReal (HSub.hSub (↑f b) (Function.leftLim (↑f) a)))
    -/
    symm
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt b a
      ⊢ Eq (ENNReal.ofReal (HSub.hSub (↑f b) (Function.leftLim (↑f) a))) 0
    -/
    simp [ENNReal.ofReal_eq_zero, f.mono.le_leftLim hab]
    /-
      🎉 no goals
    -/


@[simp]
theorem measure_Ioo {a b : ℝ} : f.measure (Ioo a b) = ofReal (leftLim f b - f a) := by
  /-
    f : StieltjesFunction
    a b : Real
    ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
  -/
  rcases le_or_lt b a with (hab | hab)
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le b a
      ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
  · simp only [hab, measure_empty, Ioo_eq_empty, not_lt]
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le b a
      ⊢ Eq 0 (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑f) b) (↑f a)))
    -/
    symm
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le b a
      ⊢ Eq (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑f) b) (↑f a))) 0
    -/
    simp [ENNReal.ofReal_eq_zero, f.mono.leftLim_le hab]
    /-
      🎉 no goals
    -/
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt a b
      ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
  · have A : Disjoint (Ioo a b) {b} := by simp
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt a b
      A : Disjoint (Set.Ioo a b) (Singleton.singleton b)
      ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
    have D : f b - f a = f b - leftLim f b + (leftLim f b - f a) := by abel
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt a b
      A : Disjoint (Set.Ioo a b) (Singleton.singleton b)
      D : Eq (HSub.hSub (↑f b) (↑f a)) (HAdd.hAdd (HSub.hSub (↑f b) (Function.leftLi …
      ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
    have := f.measure_Ioc a b
    simp only [← Ioo_union_Icc_eq_Ioc hab le_rfl, measure_singleton,
      measure_union A (measurableSet_singleton b), Icc_self] at this
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt a b
      A : Disjoint (Set.Ioo a b) (Singleton.singleton b)
      D : Eq (HSub.hSub (↑f b) (↑f a)) (HAdd.hAdd (HSub.hSub (↑f b) (Function.leftLi …
      this : Eq (HAdd.hAdd (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (↑f  …
      ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
    rw [D, ENNReal.ofReal_add, add_comm] at this
      /-
        case inr
        f : StieltjesFunction
        a b : Real
        hab : LT.lt a b
        A : Disjoint (Set.Ioo a b) (Singleton.singleton b)
        D : Eq (HSub.hSub (↑f b) (↑f a)) (HAdd.hAdd (HSub.hSub (↑f b) (Function.leftLi …
        this : Eq (HAdd.hAdd (ENNReal.ofReal (HSub.hSub (↑f b) (Function.leftLim (↑f)  …
        ⊢ Eq (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
      -/
    · simpa only [ENNReal.add_right_inj ENNReal.ofReal_ne_top]
      /-
        🎉 no goals
      -/
      /-
        case inr.hp
        f : StieltjesFunction
        a b : Real
        hab : LT.lt a b
        A : Disjoint (Set.Ioo a b) (Singleton.singleton b)
        D : Eq (HSub.hSub (↑f b) (↑f a)) (HAdd.hAdd (HSub.hSub (↑f b) (Function.leftLi …
        this : Eq (HAdd.hAdd (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (↑f  …
        ⊢ LE.le 0 (HSub.hSub (↑f b) (Function.leftLim (↑f) b))
      -/
    · simp only [f.mono.leftLim_le le_rfl, sub_nonneg]
      /-
        🎉 no goals
      -/
      /-
        case inr.hq
        f : StieltjesFunction
        a b : Real
        hab : LT.lt a b
        A : Disjoint (Set.Ioo a b) (Singleton.singleton b)
        D : Eq (HSub.hSub (↑f b) (↑f a)) (HAdd.hAdd (HSub.hSub (↑f b) (Function.leftLi …
        this : Eq (HAdd.hAdd (f.measure (Set.Ioo a b)) (ENNReal.ofReal (HSub.hSub (↑f  …
        ⊢ LE.le 0 (HSub.hSub (Function.leftLim (↑f) b) (↑f a))
      -/
    · simp only [f.mono.le_leftLim hab, sub_nonneg]
      /-
        🎉 no goals
      -/


@[simp]
theorem measure_Ico (a b : ℝ) : f.measure (Ico a b) = ofReal (leftLim f b - leftLim f a) := by
  /-
    f : StieltjesFunction
    a b : Real
    ⊢ Eq (f.measure (Set.Ico a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
  -/
  rcases le_or_lt b a with (hab | hab)
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le b a
      ⊢ Eq (f.measure (Set.Ico a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
  · simp only [hab, measure_empty, Ico_eq_empty, not_lt]
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le b a
      ⊢ Eq 0 (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑f) b) (Function.leftLim  …
    -/
    symm
    /-
      case inl
      f : StieltjesFunction
      a b : Real
      hab : LE.le b a
      ⊢ Eq (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑f) b) (Function.leftLim (↑ …
    -/
    simp [ENNReal.ofReal_eq_zero, f.mono.leftLim hab]
    /-
      🎉 no goals
    -/
    /-
      case inr
      f : StieltjesFunction
      a b : Real
      hab : LT.lt a b
      ⊢ Eq (f.measure (Set.Ico a b)) (ENNReal.ofReal (HSub.hSub (Function.leftLim (↑ …
    -/
  · have A : Disjoint {a} (Ioo a b) := by simp
    simp [← Icc_union_Ioo_eq_Ico le_rfl hab, -singleton_union, hab.ne, f.mono.leftLim_le,
      measure_union A measurableSet_Ioo, f.mono.le_leftLim hab, ← ENNReal.ofReal_add]


theorem measure_Iic {l : ℝ} (hf : Tendsto f atBot (𝓝 l)) (x : ℝ) :
    f.measure (Iic x) = ofReal (f x - l) := by
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    x : Real
    ⊢ Eq (f.measure (Set.Iic x)) (ENNReal.ofReal (HSub.hSub (↑f x) l))
  -/
  refine tendsto_nhds_unique (tendsto_measure_Ioc_atBot _ _) ?_
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    x : Real
    ⊢ Filter.Tendsto (fun x_1 => f.measure (Set.Ioc x_1 x)) Filter.atBot (nhds (EN …
  -/
  simp_rw [measure_Ioc]
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    x : Real
    ⊢ Filter.Tendsto (fun x_1 => ENNReal.ofReal (HSub.hSub (↑f x) (↑f x_1))) Filte …
  -/
  exact ENNReal.tendsto_ofReal (Tendsto.const_sub _ hf)
  /-
    🎉 no goals
  -/


lemma measure_Iio {l : ℝ} (hf : Tendsto f atBot (𝓝 l)) (x : ℝ) :
    f.measure (Iio x) = ofReal (leftLim f x - l) := by
  rw [← Iic_diff_right, measure_diff _ (nullMeasurableSet_singleton x), measure_singleton,
    f.measure_Iic hf, ← ofReal_sub _ (sub_nonneg.mpr <| Monotone.leftLim_le f.mono' le_rfl)]
        /-
          f : StieltjesFunction
          l : Real
          hf : Filter.Tendsto (↑f) Filter.atBot (nhds l)
          x : Real
          ⊢ Eq (ENNReal.ofReal (HSub.hSub (HSub.hSub (↑f x) l) (HSub.hSub (↑f x) (Functi …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
    <;> simp
        /-
          🎉 no goals
        -/


theorem measure_Ici {l : ℝ} (hf : Tendsto f atTop (𝓝 l)) (x : ℝ) :
    f.measure (Ici x) = ofReal (l - leftLim f x) := by
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    ⊢ Eq (f.measure (Set.Ici x)) (ENNReal.ofReal (HSub.hSub l (Function.leftLim (↑ …
  -/
  refine tendsto_nhds_unique (tendsto_measure_Ico_atTop _ _) ?_
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    ⊢ Filter.Tendsto (fun x_1 => f.measure (Set.Ico x x_1)) Filter.atTop (nhds (EN …
  -/
  simp_rw [measure_Ico]
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    ⊢ Filter.Tendsto (fun x_1 => ENNReal.ofReal (HSub.hSub (Function.leftLim (↑f)  …
  -/
  refine ENNReal.tendsto_ofReal (Tendsto.sub_const ?_ _)
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    ⊢ Filter.Tendsto (Function.leftLim ↑f) Filter.atTop (nhds l)
  -/
  have h_le1 : ∀ x, f (x - 1) ≤ leftLim f x := fun x => Monotone.le_leftLim f.mono (sub_one_lt x)
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    h_le1 : ∀ (x : Real), LE.le (↑f (HSub.hSub x 1)) (Function.leftLim (↑f) x)
    ⊢ Filter.Tendsto (Function.leftLim ↑f) Filter.atTop (nhds l)
  -/
  have h_le2 : ∀ x, leftLim f x ≤ f x := fun x => Monotone.leftLim_le f.mono le_rfl
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    h_le1 : ∀ (x : Real), LE.le (↑f (HSub.hSub x 1)) (Function.leftLim (↑f) x)
    h_le2 : ∀ (x : Real), LE.le (Function.leftLim (↑f) x) (↑f x)
    ⊢ Filter.Tendsto (Function.leftLim ↑f) Filter.atTop (nhds l)
  -/
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le (hf.comp ?_) hf h_le1 h_le2
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    h_le1 : ∀ (x : Real), LE.le (↑f (HSub.hSub x 1)) (Function.leftLim (↑f) x)
    h_le2 : ∀ (x : Real), LE.le (Function.leftLim (↑f) x) (↑f x)
    ⊢ Filter.Tendsto (fun i => HSub.hSub i 1) Filter.atTop Filter.atTop
  -/
  rw [tendsto_atTop_atTop]
  /-
    f : StieltjesFunction
    l : Real
    hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
    x : Real
    h_le1 : ∀ (x : Real), LE.le (↑f (HSub.hSub x 1)) (Function.leftLim (↑f) x)
    h_le2 : ∀ (x : Real), LE.le (Function.leftLim (↑f) x) (↑f x)
    ⊢ ∀ (b : Real), Exists fun i => ∀ (a : Real), LE.le i a → LE.le b (HSub.hSub a …
  -/
  exact fun y => ⟨y + 1, fun z hyz => by rwa [le_sub_iff_add_le]⟩
  /-
    🎉 no goals
  -/


lemma measure_Ioi {l : ℝ} (hf : Tendsto f atTop (𝓝 l)) (x : ℝ) :
    f.measure (Ioi x) = ofReal (l - f x) := by
  rw [← Ici_diff_left, measure_diff _ (nullMeasurableSet_singleton x), measure_singleton,
    f.measure_Ici hf, ← ofReal_sub _ (sub_nonneg.mpr <| Monotone.leftLim_le f.mono' le_rfl)]
        /-
          f : StieltjesFunction
          l : Real
          hf : Filter.Tendsto (↑f) Filter.atTop (nhds l)
          x : Real
          ⊢ Eq (ENNReal.ofReal (HSub.hSub (HSub.hSub l (Function.leftLim (↑f) x)) (HSub. …
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
    <;> simp
        /-
          🎉 no goals
        -/


lemma measure_Ioi_of_tendsto_atTop_atTop (hf : Tendsto f atTop atTop) (x : ℝ) :
    f.measure (Ioi x) = ∞ := by
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atTop Filter.atTop
    x : Real
    ⊢ Eq (f.measure (Set.Ioi x)) Top.top
  -/
  refine ENNReal.eq_top_of_forall_nnreal_le fun r ↦ ?_
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atTop Filter.atTop
    x : Real
    r : NNReal
    ⊢ LE.le (↑r) (f.measure (Set.Ioi x))
  -/
  obtain ⟨N, hN⟩ := eventually_atTop.mp (tendsto_atTop.mp hf (r + f x))
  exact (f.measure_Ioc x (max x N) ▸ ENNReal.coe_nnreal_eq r ▸ (ENNReal.ofReal_le_ofReal <|
    le_tsub_of_add_le_right <| hN _ (le_max_right x N))).trans (measure_mono Ioc_subset_Ioi_self)


lemma measure_Ici_of_tendsto_atTop_atTop (hf : Tendsto f atTop atTop) (x : ℝ) :
    f.measure (Ici x) = ∞ := by
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atTop Filter.atTop
    x : Real
    ⊢ Eq (f.measure (Set.Ici x)) Top.top
  -/
  rw [← top_le_iff, ← f.measure_Ioi_of_tendsto_atTop_atTop hf x]
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atTop Filter.atTop
    x : Real
    ⊢ LE.le (f.measure (Set.Ioi x)) (f.measure (Set.Ici x))
  -/
  exact measure_mono Ioi_subset_Ici_self
  /-
    🎉 no goals
  -/


lemma measure_Iic_of_tendsto_atBot_atBot (hf : Tendsto f atBot atBot) (x : ℝ) :
    f.measure (Iic x) = ∞ := by
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atBot Filter.atBot
    x : Real
    ⊢ Eq (f.measure (Set.Iic x)) Top.top
  -/
  refine ENNReal.eq_top_of_forall_nnreal_le fun r ↦ ?_
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atBot Filter.atBot
    x : Real
    r : NNReal
    ⊢ LE.le (↑r) (f.measure (Set.Iic x))
  -/
  obtain ⟨N, hN⟩ := eventually_atBot.mp (tendsto_atBot.mp hf (f x - r))
  exact (f.measure_Ioc (min x N) x ▸ ENNReal.coe_nnreal_eq r ▸ (ENNReal.ofReal_le_ofReal <|
    le_sub_comm.mp <| hN _ (min_le_right x N))).trans (measure_mono Ioc_subset_Iic_self)


lemma measure_Iio_of_tendsto_atBot_atBot (hf : Tendsto f atBot atBot) (x : ℝ) :
    f.measure (Iio x) = ∞ := by
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atBot Filter.atBot
    x : Real
    ⊢ Eq (f.measure (Set.Iio x)) Top.top
  -/
  rw [← top_le_iff, ← f.measure_Iic_of_tendsto_atBot_atBot hf (x - 1)]
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atBot Filter.atBot
    x : Real
    ⊢ LE.le (f.measure (Set.Iic (HSub.hSub x 1))) (f.measure (Set.Iio x))
  -/
  exact measure_mono <| Set.Iic_subset_Iio.mpr <| sub_one_lt x
  /-
    🎉 no goals
  -/


theorem measure_univ {l u : ℝ} (hfl : Tendsto f atBot (𝓝 l)) (hfu : Tendsto f atTop (𝓝 u)) :
    f.measure univ = ofReal (u - l) := by
  /-
    f : StieltjesFunction
    l u : Real
    hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    hfu : Filter.Tendsto (↑f) Filter.atTop (nhds u)
    ⊢ Eq (f.measure Set.univ) (ENNReal.ofReal (HSub.hSub u l))
  -/
  refine tendsto_nhds_unique (tendsto_measure_Iic_atTop _) ?_
  /-
    f : StieltjesFunction
    l u : Real
    hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    hfu : Filter.Tendsto (↑f) Filter.atTop (nhds u)
    ⊢ Filter.Tendsto (fun x => f.measure (Set.Iic x)) Filter.atTop (nhds (ENNReal. …
  -/
  simp_rw [measure_Iic f hfl]
  /-
    f : StieltjesFunction
    l u : Real
    hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    hfu : Filter.Tendsto (↑f) Filter.atTop (nhds u)
    ⊢ Filter.Tendsto (fun x => ENNReal.ofReal (HSub.hSub (↑f x) l)) Filter.atTop ( …
  -/
  exact ENNReal.tendsto_ofReal (Tendsto.sub_const hfu _)
  /-
    🎉 no goals
  -/


lemma measure_univ_of_tendsto_atTop_atTop (hf : Tendsto f atTop atTop) :
    f.measure univ = ∞ := by
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atTop Filter.atTop
    ⊢ Eq (f.measure Set.univ) Top.top
  -/
  rw [← top_le_iff, ← f.measure_Ioi_of_tendsto_atTop_atTop hf 0]
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atTop Filter.atTop
    ⊢ LE.le (f.measure (Set.Ioi 0)) (f.measure Set.univ)
  -/
  exact measure_mono (subset_univ _)
  /-
    🎉 no goals
  -/


lemma measure_univ_of_tendsto_atBot_atBot (hf : Tendsto f atBot atBot) :
    f.measure univ = ∞ := by
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atBot Filter.atBot
    ⊢ Eq (f.measure Set.univ) Top.top
  -/
  rw [← top_le_iff, ← f.measure_Iio_of_tendsto_atBot_atBot hf 0]
  /-
    f : StieltjesFunction
    hf : Filter.Tendsto (↑f) Filter.atBot Filter.atBot
    ⊢ LE.le (f.measure (Set.Iio 0)) (f.measure Set.univ)
  -/
  exact measure_mono (subset_univ _)
  /-
    🎉 no goals
  -/


lemma isFiniteMeasure {l u : ℝ} (hfl : Tendsto f atBot (𝓝 l)) (hfu : Tendsto f atTop (𝓝 u)) :
                                     /-
                                       f : StieltjesFunction
                                       l u : Real
                                       hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
                                       hfu : Filter.Tendsto (↑f) Filter.atTop (nhds u)
                                       ⊢ LT.lt (f.measure Set.univ) Top.top
                                     -/
    IsFiniteMeasure f.measure := ⟨by simp [f.measure_univ hfl hfu]⟩
                                     /-
                                       🎉 no goals
                                     -/


lemma isProbabilityMeasure (hf_bot : Tendsto f atBot (𝓝 0)) (hf_top : Tendsto f atTop (𝓝 1)) :
                                          /-
                                            f : StieltjesFunction
                                            hf_bot : Filter.Tendsto (↑f) Filter.atBot (nhds 0)
                                            hf_top : Filter.Tendsto (↑f) Filter.atTop (nhds 1)
                                            ⊢ Eq (f.measure Set.univ) 1
                                          -/
    IsProbabilityMeasure f.measure := ⟨by simp [f.measure_univ hf_bot hf_top]⟩
                                          /-
                                            🎉 no goals
                                          -/


instance instIsLocallyFiniteMeasure : IsLocallyFiniteMeasure f.measure :=
                                                   /-
                                                     f : StieltjesFunction
                                                     x : Real
                                                     ⊢ LT.lt (HSub.hSub x 1) x
                                                   -/
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  ⟨fun x => ⟨Ioo (x - 1) (x + 1), Ioo_mem_nhds (by linarith) (by linarith), by simp⟩⟩
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma eq_of_measure_of_tendsto_atBot (g : StieltjesFunction) {l : ℝ}
    (hfg : f.measure = g.measure) (hfl : Tendsto f atBot (𝓝 l)) (hgl : Tendsto g atBot (𝓝 l)) :
    f = g := by
  /-
    f g : StieltjesFunction
    l : Real
    hfg : Eq f.measure g.measure
    hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
    ⊢ Eq f g
  -/
  ext x
  /-
    case h
    f g : StieltjesFunction
    l : Real
    hfg : Eq f.measure g.measure
    hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
    x : Real
    ⊢ Eq (↑f x) (↑g x)
  -/
  have hf := measure_Iic f hfl x
  /-
    case h
    f g : StieltjesFunction
    l : Real
    hfg : Eq f.measure g.measure
    hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
    hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
    x : Real
    hf : Eq (f.measure (Set.Iic x)) (ENNReal.ofReal (HSub.hSub (↑f x) l))
    ⊢ Eq (↑f x) (↑g x)
  -/
  rw [hfg, measure_Iic g hgl x, ENNReal.ofReal_eq_ofReal_iff, eq_comm] at hf
    /-
      case h
      f g : StieltjesFunction
      l : Real
      hfg : Eq f.measure g.measure
      hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
      hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
      x : Real
      hf : Eq (HSub.hSub (↑f x) l) (HSub.hSub (↑g x) l)
      ⊢ Eq (↑f x) (↑g x)
    -/
  · simpa using hf
    /-
      🎉 no goals
    -/
    /-
      case h.hp
      f g : StieltjesFunction
      l : Real
      hfg : Eq f.measure g.measure
      hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
      hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
      x : Real
      hf : Eq (ENNReal.ofReal (HSub.hSub (↑g x) l)) (ENNReal.ofReal (HSub.hSub (↑f x …
      ⊢ LE.le 0 (HSub.hSub (↑g x) l)
    -/
  · rw [sub_nonneg]
    /-
      case h.hp
      f g : StieltjesFunction
      l : Real
      hfg : Eq f.measure g.measure
      hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
      hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
      x : Real
      hf : Eq (ENNReal.ofReal (HSub.hSub (↑g x) l)) (ENNReal.ofReal (HSub.hSub (↑f x …
      ⊢ LE.le l (↑g x)
    -/
    exact Monotone.le_of_tendsto g.mono hgl x
    /-
      🎉 no goals
    -/
    /-
      case h.hq
      f g : StieltjesFunction
      l : Real
      hfg : Eq f.measure g.measure
      hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
      hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
      x : Real
      hf : Eq (ENNReal.ofReal (HSub.hSub (↑g x) l)) (ENNReal.ofReal (HSub.hSub (↑f x …
      ⊢ LE.le 0 (HSub.hSub (↑f x) l)
    -/
  · rw [sub_nonneg]
    /-
      case h.hq
      f g : StieltjesFunction
      l : Real
      hfg : Eq f.measure g.measure
      hfl : Filter.Tendsto (↑f) Filter.atBot (nhds l)
      hgl : Filter.Tendsto (↑g) Filter.atBot (nhds l)
      x : Real
      hf : Eq (ENNReal.ofReal (HSub.hSub (↑g x) l)) (ENNReal.ofReal (HSub.hSub (↑f x …
      ⊢ LE.le l (↑f x)
    -/
    exact Monotone.le_of_tendsto f.mono hfl x
    /-
      🎉 no goals
    -/


lemma eq_of_measure_of_eq (g : StieltjesFunction) {y : ℝ}
    (hfg : f.measure = g.measure) (hy : f y = g y) :
    f = g := by
  /-
    f g : StieltjesFunction
    y : Real
    hfg : Eq f.measure g.measure
    hy : Eq (↑f y) (↑g y)
    ⊢ Eq f g
  -/
  ext x
  cases le_total x y with
  | inl hxy =>
    have hf := measure_Ioc f x y
    rw [hfg, measure_Ioc g x y, ENNReal.ofReal_eq_ofReal_iff, eq_comm, hy] at hf
    · simpa using hf
    · rw [sub_nonneg]
      exact g.mono hxy
    · rw [sub_nonneg]
      exact f.mono hxy
  | inr hxy =>
    have hf := measure_Ioc f y x
    rw [hfg, measure_Ioc g y x, ENNReal.ofReal_eq_ofReal_iff, eq_comm, hy] at hf
    · simpa using hf
    · rw [sub_nonneg]
      exact g.mono hxy
    · rw [sub_nonneg]
      exact f.mono hxy


@[simp]
lemma measure_zero : StieltjesFunction.measure 0 = 0 :=
                             /-
                               ⊢ ∀ ⦃a b : Real⦄, LT.lt a b → Eq ((StieltjesFunction.measure 0) (Set.Ioc a b)) …
                             -/
  Measure.ext_of_Ioc _ _ (by simp)
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma measure_const (c : ℝ) : (StieltjesFunction.const c).measure = 0 :=
                             /-
                               c : Real
                               ⊢ ∀ ⦃a b : Real⦄, LT.lt a b → Eq ((StieltjesFunction.const c).measure (Set.Ioc …
                             -/
  Measure.ext_of_Ioc _ _ (by simp)
                             /-
                               🎉 no goals
                             -/


@[simp]
lemma measure_add (f g : StieltjesFunction) : (f + g).measure = f.measure + g.measure := by
  /-
    f g : StieltjesFunction
    ⊢ Eq (HAdd.hAdd f g).measure (HAdd.hAdd f.measure g.measure)
  -/
  refine Measure.ext_of_Ioc _ _ (fun a b h ↦ ?_)
  /-
    f g : StieltjesFunction
    a b : Real
    h : LT.lt a b
    ⊢ Eq ((HAdd.hAdd f g).measure (Set.Ioc a b)) ((HAdd.hAdd f.measure g.measure)  …
  -/
  simp only [measure_Ioc, add_apply, Measure.coe_add, Pi.add_apply]
  /-
    f g : StieltjesFunction
    a b : Real
    h : LT.lt a b
    ⊢ Eq (ENNReal.ofReal (HSub.hSub (HAdd.hAdd (↑f b) (↑g b)) (HAdd.hAdd (↑f a) (↑ …
  -/
  rw [← ENNReal.ofReal_add (sub_nonneg_of_le (f.mono h.le)) (sub_nonneg_of_le (g.mono h.le))]
  /-
    f g : StieltjesFunction
    a b : Real
    h : LT.lt a b
    ⊢ Eq (ENNReal.ofReal (HSub.hSub (HAdd.hAdd (↑f b) (↑g b)) (HAdd.hAdd (↑f a) (↑ …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


@[simp]
lemma measure_smul (c : ℝ≥0) (f : StieltjesFunction) : (c • f).measure = c • f.measure := by
  /-
    c : NNReal
    f : StieltjesFunction
    ⊢ Eq (HSMul.hSMul c f).measure (HSMul.hSMul c f.measure)
  -/
  refine Measure.ext_of_Ioc _ _ (fun a b _ ↦ ?_)
  /-
    c : NNReal
    f : StieltjesFunction
    a b : Real
    x✝ : LT.lt a b
    ⊢ Eq ((HSMul.hSMul c f).measure (Set.Ioc a b)) ((HSMul.hSMul c f.measure) (Set …
  -/
  simp only [measure_Ioc, Measure.smul_apply]
  /-
    c : NNReal
    f : StieltjesFunction
    a b : Real
    x✝ : LT.lt a b
    ⊢ Eq (ENNReal.ofReal (HSub.hSub (↑(HSMul.hSMul c f) b) (↑(HSMul.hSMul c f) a)) …
  -/
  change ofReal (c * f b - c * f a) = c • ofReal (f b - f a)
  /-
    c : NNReal
    f : StieltjesFunction
    a b : Real
    x✝ : LT.lt a b
    ⊢ Eq (ENNReal.ofReal (HSub.hSub (HMul.hMul (↑c) (↑f b)) (HMul.hMul (↑c) (↑f a) …
  -/
  rw [← _root_.mul_sub, ENNReal.ofReal_mul zero_le_coe, ofReal_coe_nnreal, ← smul_eq_mul]
  /-
    c : NNReal
    f : StieltjesFunction
    a b : Real
    x✝ : LT.lt a b
    ⊢ Eq (HSMul.hSMul (↑c) (ENNReal.ofReal (HSub.hSub (↑f b) (↑f a)))) (HSMul.hSMu …
  -/
  rfl
  /-
    🎉 no goals
  -/


