theorem continuous_right_toIcoMod : ContinuousWithinAt (toIcoMod hp a) (Ici x) x := by
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    ⊢ ContinuousWithinAt (toIcoMod hp a) (Set.Ici x) x
  -/
  intro s h
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds (toIcoMod hp a x)) s
    ⊢ Membership.mem (Filter.map (toIcoMod hp a) (nhdsWithin x (Set.Ici x))) s
  -/
  rw [Filter.mem_map, mem_nhdsWithin_iff_exists_mem_nhds_inter]
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds (toIcoMod hp a x)) s
    ⊢ Exists fun u => And (Membership.mem (nhds x) u) (HasSubset.Subset (Inter.int …
  -/
  haveI : Nontrivial 𝕜 := ⟨⟨0, p, hp.ne⟩⟩
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    h : Membership.mem (nhds (toIcoMod hp a x)) s
    this : Nontrivial 𝕜
    ⊢ Exists fun u => And (Membership.mem (nhds x) u) (HasSubset.Subset (Inter.int …
  -/
  simp_rw [mem_nhds_iff_exists_Ioo_subset] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    this : Nontrivial 𝕜
    h : Exists fun l => Exists fun u => And (Membership.mem (Set.Ioo l u) (toIcoMo …
    ⊢ Exists fun u => And (Exists fun l => Exists fun u_1 => And (Membership.mem ( …
  -/
  obtain ⟨l, u, hxI, hIs⟩ := h
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    this : Nontrivial 𝕜
    l u : 𝕜
    hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
    hIs : HasSubset.Subset (Set.Ioo l u) s
    ⊢ Exists fun u => And (Exists fun l => Exists fun u_1 => And (Membership.mem ( …
  -/
  let d := toIcoDiv hp a x • p
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    this : Nontrivial 𝕜
    l u : 𝕜
    hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
    hIs : HasSubset.Subset (Set.Ioo l u) s
    d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
    ⊢ Exists fun u => And (Exists fun l => Exists fun u_1 => And (Membership.mem ( …
  -/
  have hd := toIcoMod_mem_Ico hp a x
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    this : Nontrivial 𝕜
    l u : 𝕜
    hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
    hIs : HasSubset.Subset (Set.Ioo l u) s
    d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
    hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
    ⊢ Exists fun u => And (Exists fun l => Exists fun u_1 => And (Membership.mem ( …
  -/
  simp_rw [subset_def, mem_inter_iff]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    s : Set 𝕜
    this : Nontrivial 𝕜
    l u : 𝕜
    hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
    hIs : HasSubset.Subset (Set.Ioo l u) s
    d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
    hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
    ⊢ Exists fun u => And (Exists fun l => Exists fun u_1 => And (Membership.mem ( …
  -/
  refine ⟨_, ⟨l + d, min (a + p) u + d, ?_, fun x => id⟩, fun y => ?_⟩ <;>
    /-
      case intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      a x : 𝕜
      s : Set 𝕜
      this : Nontrivial 𝕜
      l u : 𝕜
      hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
      hIs : HasSubset.Subset (Set.Ioo l u) s
      d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
      hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
      ⊢ Membership.mem (Set.Ioo (HAdd.hAdd l d) (HAdd.hAdd (Min.min (HAdd.hAdd a p)  …
    -/
    simp_rw [← sub_mem_Ioo_iff_left, mem_Ioo, lt_min_iff]
    /-
      case intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      a x : 𝕜
      s : Set 𝕜
      this : Nontrivial 𝕜
      l u : 𝕜
      hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
      hIs : HasSubset.Subset (Set.Ioo l u) s
      d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
      hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
      ⊢ And (LT.lt l (HSub.hSub x d)) (And (LT.lt (HSub.hSub x d) (HAdd.hAdd a p)) ( …
    -/
  · exact ⟨hxI.1, hd.2, hxI.2⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      a x : 𝕜
      s : Set 𝕜
      this : Nontrivial 𝕜
      l u : 𝕜
      hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
      hIs : HasSubset.Subset (Set.Ioo l u) s
      d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
      hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
      y : 𝕜
      ⊢ And (And (LT.lt l (HSub.hSub y d)) (And (LT.lt (HSub.hSub y d) (HAdd.hAdd a  …
    -/
  · rintro ⟨h, h'⟩
    /-
      case intro.intro.intro.refine_2.intro
      𝕜 : Type u_1
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      a x : 𝕜
      s : Set 𝕜
      this : Nontrivial 𝕜
      l u : 𝕜
      hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
      hIs : HasSubset.Subset (Set.Ioo l u) s
      d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
      hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
      y : 𝕜
      h : And (LT.lt l (HSub.hSub y d)) (And (LT.lt (HSub.hSub y d) (HAdd.hAdd a p)) …
      h' : Membership.mem (Set.Ici x) y
      ⊢ Membership.mem (Set.preimage (toIcoMod hp a) s) y
    -/
    apply hIs
    /-
      case intro.intro.intro.refine_2.intro.a
      𝕜 : Type u_1
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      a x : 𝕜
      s : Set 𝕜
      this : Nontrivial 𝕜
      l u : 𝕜
      hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
      hIs : HasSubset.Subset (Set.Ioo l u) s
      d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
      hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
      y : 𝕜
      h : And (LT.lt l (HSub.hSub y d)) (And (LT.lt (HSub.hSub y d) (HAdd.hAdd a p)) …
      h' : Membership.mem (Set.Ici x) y
      ⊢ Membership.mem (Set.Ioo l u) (toIcoMod hp a y)
    -/
    rw [← toIcoMod_sub_zsmul, (toIcoMod_eq_self _).2]
    /-
      case intro.intro.intro.refine_2.intro.a
      𝕜 : Type u_1
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      a x : 𝕜
      s : Set 𝕜
      this : Nontrivial 𝕜
      l u : 𝕜
      hxI : Membership.mem (Set.Ioo l u) (toIcoMod hp a x)
      hIs : HasSubset.Subset (Set.Ioo l u) s
      d : 𝕜 := HSMul.hSMul (toIcoDiv hp a x) p
      hd : Membership.mem (Set.Ico a (HAdd.hAdd a p)) (toIcoMod hp a x)
      y : 𝕜
      h : And (LT.lt l (HSub.hSub y d)) (And (LT.lt (HSub.hSub y d) (HAdd.hAdd a p)) …
      h' : Membership.mem (Set.Ici x) y
      ⊢ Membership.mem (Set.Ioo l u) (HSub.hSub y (HSMul.hSMul ?intro.intro.intro.re …
    -/
    exacts [⟨h.1, h.2.2⟩, ⟨hd.1.trans (sub_le_sub_right h' _), h.2.1⟩]
    /-
      🎉 no goals
    -/


theorem continuous_left_toIocMod : ContinuousWithinAt (toIocMod hp a) (Iic x) x := by
  rw [(funext fun y => Eq.trans (by rw [neg_neg]) <| toIocMod_neg _ _ _ :
      toIocMod hp a = (fun x => p - x) ∘ toIcoMod hp (-a) ∘ Neg.neg)]
  -- Porting note: added
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    a x : 𝕜
    ⊢ ContinuousWithinAt (Function.comp (fun x => HSub.hSub p x) (Function.comp (t …
  -/
  have : ContinuousNeg 𝕜 := TopologicalAddGroup.toContinuousNeg
  exact
    (continuous_sub_left _).continuousAt.comp_continuousWithinAt <|
      (continuous_right_toIcoMod _ _ _).comp continuous_neg.continuousWithinAt fun y => neg_le_neg


theorem toIcoMod_eventuallyEq_toIocMod (hx : (x : 𝕜 ⧸ zmultiples p) ≠ a) :
    toIcoMod hp a =ᶠ[𝓝 x] toIocMod hp a :=
  IsOpen.mem_nhds
      (by
        /-
          𝕜 : Type u_1
          inst✝³ : LinearOrderedAddCommGroup 𝕜
          inst✝² : Archimedean 𝕜
          inst✝¹ : TopologicalSpace 𝕜
          inst✝ : OrderTopology 𝕜
          p : 𝕜
          hp : LT.lt 0 p
          a x : 𝕜
          hx : Ne ↑x ↑a
          ⊢ IsOpen (setOf fun x => (fun x => Eq (toIcoMod hp a x) (toIocMod hp a x)) x)
        -/
        rw [Ico_eq_locus_Ioc_eq_iUnion_Ioo]
        /-
          𝕜 : Type u_1
          inst✝³ : LinearOrderedAddCommGroup 𝕜
          inst✝² : Archimedean 𝕜
          inst✝¹ : TopologicalSpace 𝕜
          inst✝ : OrderTopology 𝕜
          p : 𝕜
          hp : LT.lt 0 p
          a x : 𝕜
          hx : Ne ↑x ↑a
          ⊢ IsOpen (Set.iUnion fun z => Set.Ioo (HAdd.hAdd a (HSMul.hSMul z p)) (HAdd.hA …
        -/
        exact isOpen_iUnion fun i => isOpen_Ioo) <|
        /-
          🎉 no goals
        -/
    (not_modEq_iff_toIcoMod_eq_toIocMod hp).1 <| not_modEq_iff_ne_mod_zmultiples.2 hx


theorem continuousAt_toIcoMod (hx : (x : 𝕜 ⧸ zmultiples p) ≠ a) : ContinuousAt (toIcoMod hp a) x :=
  let h := toIcoMod_eventuallyEq_toIocMod hp a hx
  continuousAt_iff_continuous_left_right.2 <|
    ⟨(continuous_left_toIocMod hp a x).congr_of_eventuallyEq (h.filter_mono nhdsWithin_le_nhds)
        h.eq_of_nhds,
      continuous_right_toIcoMod hp a x⟩


theorem continuousAt_toIocMod (hx : (x : 𝕜 ⧸ zmultiples p) ≠ a) : ContinuousAt (toIocMod hp a) x :=
  let h := toIcoMod_eventuallyEq_toIocMod hp a hx
  continuousAt_iff_continuous_left_right.2 <|
    ⟨continuous_left_toIocMod hp a x,
      (continuous_right_toIcoMod hp a x).congr_of_eventuallyEq
        (h.symm.filter_mono nhdsWithin_le_nhds) h.symm.eq_of_nhds⟩


/-- The "additive circle": `𝕜 ⧸ (ℤ ∙ p)`. See also `Circle` and `Real.angle`. -/
abbrev AddCircle [LinearOrderedAddCommGroup 𝕜] (p : 𝕜) :=
  𝕜 ⧸ zmultiples p


theorem coe_nsmul {n : ℕ} {x : 𝕜} : (↑(n • x) : AddCircle p) = n • (x : AddCircle p) :=
  rfl


theorem coe_zsmul {n : ℤ} {x : 𝕜} : (↑(n • x) : AddCircle p) = n • (x : AddCircle p) :=
  rfl


theorem coe_add (x y : 𝕜) : (↑(x + y) : AddCircle p) = (x : AddCircle p) + (y : AddCircle p) :=
  rfl


theorem coe_sub (x y : 𝕜) : (↑(x - y) : AddCircle p) = (x : AddCircle p) - (y : AddCircle p) :=
  rfl


theorem coe_neg {x : 𝕜} : (↑(-x) : AddCircle p) = -(x : AddCircle p) :=
  rfl


theorem coe_eq_zero_iff {x : 𝕜} : (x : AddCircle p) = 0 ↔ ∃ n : ℤ, n • p = x := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedAddCommGroup 𝕜
    p x : 𝕜
    ⊢ Iff (Eq (↑x) 0) (Exists fun n => Eq (HSMul.hSMul n p) x)
  -/
  simp [AddSubgroup.mem_zmultiples_iff]
  /-
    🎉 no goals
  -/


theorem coe_eq_zero_of_pos_iff (hp : 0 < p) {x : 𝕜} (hx : 0 < x) :
    (x : AddCircle p) = 0 ↔ ∃ n : ℕ, n • p = x := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Iff (Eq (↑x) 0) (Exists fun n => Eq (HSMul.hSMul n p) x)
  -/
  rw [coe_eq_zero_iff]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : LT.lt 0 p
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Iff (Exists fun n => Eq (HSMul.hSMul n p) x) (Exists fun n => Eq (HSMul.hSMu …
  -/
  constructor <;> rintro ⟨n, rfl⟩
  · replace hx : 0 < n := by
      contrapose! hx
      simpa only [← neg_nonneg, ← zsmul_neg, zsmul_neg'] using zsmul_nonneg hp.le (neg_nonneg.2 hx)
    /-
      case mp.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedAddCommGroup 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      n : Int
      hx : LT.lt 0 n
      ⊢ Exists fun n_1 => Eq (HSMul.hSMul n_1 p) (HSMul.hSMul n p)
    -/
    exact ⟨n.toNat, by rw [← natCast_zsmul, Int.toNat_of_nonneg hx.le]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedAddCommGroup 𝕜
      p : 𝕜
      hp : LT.lt 0 p
      n : Nat
      hx : LT.lt 0 (HSMul.hSMul n p)
      ⊢ Exists fun n_1 => Eq (HSMul.hSMul n_1 p) (HSMul.hSMul n p)
    -/
  · exact ⟨(n : ℤ), by simp⟩
    /-
      🎉 no goals
    -/


theorem coe_period : (p : AddCircle p) = 0 :=
  (QuotientAddGroup.eq_zero_iff p).2 <| mem_zmultiples p


theorem coe_add_period (x : 𝕜) : ((x + p : 𝕜) : AddCircle p) = x := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedAddCommGroup 𝕜
    p x : 𝕜
    ⊢ Eq ↑(HAdd.hAdd x p) ↑x
  -/
  rw [coe_add, ← eq_sub_iff_add_eq', sub_self, coe_period]
  /-
    🎉 no goals
  -/


@[continuity, nolint unusedArguments]
protected theorem continuous_mk' [TopologicalSpace 𝕜] :
    Continuous (QuotientAddGroup.mk' (zmultiples p) : 𝕜 → AddCircle p) :=
  continuous_coinduced_rng


/-- The equivalence between `AddCircle p` and the half-open interval `[a, a + p)`, whose inverse
is the natural quotient map. -/
def equivIco : AddCircle p ≃ Ico a (a + p) :=
  QuotientAddGroup.equivIcoMod hp.out a


/-- The equivalence between `AddCircle p` and the half-open interval `(a, a + p]`, whose inverse
is the natural quotient map. -/
def equivIoc : AddCircle p ≃ Ioc a (a + p) :=
  QuotientAddGroup.equivIocMod hp.out a


/-- Given a function on `𝕜`, return the unique function on `AddCircle p` agreeing with `f` on
`[a, a + p)`. -/
def liftIco (f : 𝕜 → B) : AddCircle p → B :=
  restrict _ f ∘ AddCircle.equivIco p a


/-- Given a function on `𝕜`, return the unique function on `AddCircle p` agreeing with `f` on
`(a, a + p]`. -/
def liftIoc (f : 𝕜 → B) : AddCircle p → B :=
  restrict _ f ∘ AddCircle.equivIoc p a


theorem coe_eq_coe_iff_of_mem_Ico {x y : 𝕜} (hx : x ∈ Ico a (a + p)) (hy : y ∈ Ico a (a + p)) :
    (x : AddCircle p) = y ↔ x = y := by
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    x y : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    hy : Membership.mem (Set.Ico a (HAdd.hAdd a p)) y
    ⊢ Iff (Eq ↑x ↑y) (Eq x y)
  -/
  refine ⟨fun h => ?_, by tauto⟩
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    x y : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    hy : Membership.mem (Set.Ico a (HAdd.hAdd a p)) y
    h : Eq ↑x ↑y
    ⊢ Eq x y
  -/
  suffices (⟨x, hx⟩ : Ico a (a + p)) = ⟨y, hy⟩ by exact Subtype.mk.inj this
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    x y : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    hy : Membership.mem (Set.Ico a (HAdd.hAdd a p)) y
    h : Eq ↑x ↑y
    ⊢ Eq ⟨x, hx⟩ ⟨y, hy⟩
  -/
  apply_fun equivIco p a at h
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    x y : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    hy : Membership.mem (Set.Ico a (HAdd.hAdd a p)) y
    h : Eq ((AddCircle.equivIco p a) ↑x) ((AddCircle.equivIco p a) ↑y)
    ⊢ Eq ⟨x, hx⟩ ⟨y, hy⟩
  -/
  rw [← (equivIco p a).right_inv ⟨x, hx⟩, ← (equivIco p a).right_inv ⟨y, hy⟩]
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    x y : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    hy : Membership.mem (Set.Ico a (HAdd.hAdd a p)) y
    h : Eq ((AddCircle.equivIco p a) ↑x) ((AddCircle.equivIco p a) ↑y)
    ⊢ Eq ((AddCircle.equivIco p a).toFun ((AddCircle.equivIco p a).invFun ⟨x, hx⟩) …
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem liftIco_coe_apply {f : 𝕜 → B} {x : 𝕜} (hx : x ∈ Ico a (a + p)) :
    liftIco p a f ↑x = f x := by
  have : (equivIco p a) x = ⟨x, hx⟩ := by
    rw [Equiv.apply_eq_iff_eq_symm_apply]
    rfl
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    f : 𝕜 → B
    x : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    this : Eq ((AddCircle.equivIco p a) ↑x) ⟨x, hx⟩
    ⊢ Eq (AddCircle.liftIco p a f ↑x) (f x)
  -/
  rw [liftIco, comp_apply, this]
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    f : 𝕜 → B
    x : 𝕜
    hx : Membership.mem (Set.Ico a (HAdd.hAdd a p)) x
    this : Eq ((AddCircle.equivIco p a) ↑x) ⟨x, hx⟩
    ⊢ Eq ((Set.Ico a (HAdd.hAdd a p)).restrict f ⟨x, hx⟩) (f x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem liftIoc_coe_apply {f : 𝕜 → B} {x : 𝕜} (hx : x ∈ Ioc a (a + p)) :
    liftIoc p a f ↑x = f x := by
  have : (equivIoc p a) x = ⟨x, hx⟩ := by
    rw [Equiv.apply_eq_iff_eq_symm_apply]
    rfl
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    f : 𝕜 → B
    x : 𝕜
    hx : Membership.mem (Set.Ioc a (HAdd.hAdd a p)) x
    this : Eq ((AddCircle.equivIoc p a) ↑x) ⟨x, hx⟩
    ⊢ Eq (AddCircle.liftIoc p a f ↑x) (f x)
  -/
  rw [liftIoc, comp_apply, this]
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    f : 𝕜 → B
    x : 𝕜
    hx : Membership.mem (Set.Ioc a (HAdd.hAdd a p)) x
    this : Eq ((AddCircle.equivIoc p a) ↑x) ⟨x, hx⟩
    ⊢ Eq ((Set.Ioc a (HAdd.hAdd a p)).restrict f ⟨x, hx⟩) (f x)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma eq_coe_Ico (a : AddCircle p) : ∃ b, b ∈ Ico 0 p ∧ ↑b = a := by
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝ : Archimedean 𝕜
    a : AddCircle p
    ⊢ Exists fun b => And (Membership.mem (Set.Ico 0 p) b) (Eq (↑b) a)
  -/
  let b := QuotientAddGroup.equivIcoMod hp.out 0 a
  exact ⟨b.1, by simpa only [zero_add] using b.2,
    (QuotientAddGroup.equivIcoMod hp.out 0).symm_apply_apply a⟩


lemma coe_eq_zero_iff_of_mem_Ico (ha : a ∈ Ico 0 p) :
    (a : AddCircle p) = 0 ↔ a = 0 := by
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ha : Membership.mem (Set.Ico 0 p) a
    ⊢ Iff (Eq (↑a) 0) (Eq a 0)
  -/
  have h0 : 0 ∈ Ico 0 (0 + p) := by simpa [zero_add, left_mem_Ico] using hp.out
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ha : Membership.mem (Set.Ico 0 p) a
    h0 : Membership.mem (Set.Ico 0 (HAdd.hAdd 0 p)) 0
    ⊢ Iff (Eq (↑a) 0) (Eq a 0)
  -/
  have ha' : a ∈ Ico 0 (0 + p) := by rwa [zero_add]
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ha : Membership.mem (Set.Ico 0 p) a
    h0 : Membership.mem (Set.Ico 0 (HAdd.hAdd 0 p)) 0
    ha' : Membership.mem (Set.Ico 0 (HAdd.hAdd 0 p)) a
    ⊢ Iff (Eq (↑a) 0) (Eq a 0)
  -/
  rw [← AddCircle.coe_eq_coe_iff_of_mem_Ico ha' h0, QuotientAddGroup.mk_zero]
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_equivIco_symm : Continuous (equivIco p a).symm :=
  continuous_quotient_mk'.comp continuous_subtype_val


@[continuity]
theorem continuous_equivIoc_symm : Continuous (equivIoc p a).symm :=
  continuous_quotient_mk'.comp continuous_subtype_val


theorem continuousAt_equivIco (hx : x ≠ a) : ContinuousAt (equivIco p a) x := by
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : AddCircle p
    hx : Ne x ↑a
    ⊢ ContinuousAt (⇑(AddCircle.equivIco p a)) x
  -/
  induction x using QuotientAddGroup.induction_on
  /-
    case H
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : AddCircle p
    z✝ : 𝕜
    hx : Ne ↑z✝ ↑a
    ⊢ ContinuousAt ⇑(AddCircle.equivIco p a) ↑z✝
  -/
  rw [ContinuousAt, Filter.Tendsto, QuotientAddGroup.nhds_eq, Filter.map_map]
  /-
    case H
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : AddCircle p
    z✝ : 𝕜
    hx : Ne ↑z✝ ↑a
    ⊢ LE.le (Filter.map (Function.comp (⇑(AddCircle.equivIco p a)) QuotientAddGrou …
  -/
  exact (continuousAt_toIcoMod hp.out a hx).codRestrict _
  /-
    🎉 no goals
  -/


theorem continuousAt_equivIoc (hx : x ≠ a) : ContinuousAt (equivIoc p a) x := by
  /-
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : AddCircle p
    hx : Ne x ↑a
    ⊢ ContinuousAt (⇑(AddCircle.equivIoc p a)) x
  -/
  induction x using QuotientAddGroup.induction_on
  /-
    case H
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : AddCircle p
    z✝ : 𝕜
    hx : Ne ↑z✝ ↑a
    ⊢ ContinuousAt ⇑(AddCircle.equivIoc p a) ↑z✝
  -/
  rw [ContinuousAt, Filter.Tendsto, QuotientAddGroup.nhds_eq, Filter.map_map]
  /-
    case H
    𝕜 : Type u_1
    inst✝³ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : AddCircle p
    z✝ : 𝕜
    hx : Ne ↑z✝ ↑a
    ⊢ LE.le (Filter.map (Function.comp (⇑(AddCircle.equivIoc p a)) QuotientAddGrou …
  -/
  exact (continuousAt_toIocMod hp.out a hx).codRestrict _
  /-
    🎉 no goals
  -/


/-- The quotient map `𝕜 → AddCircle p` as a partial homeomorphism. -/
@[simps] def partialHomeomorphCoe [DiscreteTopology (zmultiples p)] :
    PartialHomeomorph 𝕜 (AddCircle p) where
  toFun := (↑)
  invFun := fun x ↦ equivIco p a x
  source := Ioo a (a + p)
  target := {↑a}ᶜ
  map_source' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝⁴ : LinearOrderedAddCommGroup 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      a : 𝕜
      inst✝³ : Archimedean 𝕜
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : OrderTopology 𝕜
      x : AddCircle p
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultip …
      ⊢ ∀ ⦃x : 𝕜⦄, Membership.mem (Set.Ioo a (HAdd.hAdd a p)) x → Membership.mem (Ha …
    -/
    intro x hx hx'
    exact hx.1.ne' ((coe_eq_coe_iff_of_mem_Ico (Ioo_subset_Ico_self hx)
      (left_mem_Ico.mpr (lt_add_of_pos_right a hp.out))).mp hx')
  map_target' := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝⁴ : LinearOrderedAddCommGroup 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      a : 𝕜
      inst✝³ : Archimedean 𝕜
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : OrderTopology 𝕜
      x : AddCircle p
      inst✝ : DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmultip …
      ⊢ ∀ ⦃x : AddCircle p⦄, Membership.mem (HasCompl.compl (Singleton.singleton ↑a) …
    -/
    intro x hx
    exact (eq_left_or_mem_Ioo_of_mem_Ico (equivIco p a x).2).resolve_left
      (hx ∘ ((equivIco p a).symm_apply_apply x).symm.trans ∘ congrArg _)
  left_inv' :=
    fun x hx ↦ congrArg _ ((equivIco p a).apply_symm_apply ⟨x, Ioo_subset_Ico_self hx⟩)
  right_inv' := fun x _ ↦ (equivIco p a).symm_apply_apply x
  open_source := isOpen_Ioo
  open_target := isOpen_compl_singleton
  continuousOn_toFun := (AddCircle.continuous_mk' p).continuousOn
  continuousOn_invFun := by
    exact continuousOn_of_forall_continuousAt
      (fun _ ↦ continuousAt_subtype_val.comp ∘ continuousAt_equivIco p a)


lemma isLocalHomeomorph_coe [DiscreteTopology (zmultiples p)] [DenselyOrdered 𝕜] :
    IsLocalHomeomorph ((↑) : 𝕜 → AddCircle p) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝⁴ : Archimedean 𝕜
    inst✝³ : TopologicalSpace 𝕜
    inst✝² : OrderTopology 𝕜
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmulti …
    inst✝ : DenselyOrdered 𝕜
    ⊢ IsLocalHomeomorph QuotientAddGroup.mk
  -/
  intro a
  /-
    𝕜 : Type u_1
    inst✝⁵ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝⁴ : Archimedean 𝕜
    inst✝³ : TopologicalSpace 𝕜
    inst✝² : OrderTopology 𝕜
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmulti …
    inst✝ : DenselyOrdered 𝕜
    a : 𝕜
    ⊢ Exists fun e => And (Membership.mem e.source a) (Eq QuotientAddGroup.mk ↑e)
  -/
  obtain ⟨b, hb1, hb2⟩ := exists_between (sub_lt_self a hp.out)
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁵ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝⁴ : Archimedean 𝕜
    inst✝³ : TopologicalSpace 𝕜
    inst✝² : OrderTopology 𝕜
    inst✝¹ : DiscreteTopology (Subtype fun x => Membership.mem (AddSubgroup.zmulti …
    inst✝ : DenselyOrdered 𝕜
    a b : 𝕜
    hb1 : LT.lt (HSub.hSub a p) b
    hb2 : LT.lt b a
    ⊢ Exists fun e => And (Membership.mem e.source a) (Eq QuotientAddGroup.mk ↑e)
  -/
  exact ⟨partialHomeomorphCoe p b, ⟨hb2, lt_add_of_sub_right_lt hb1⟩, rfl⟩
  /-
    🎉 no goals
  -/


/-- The image of the closed-open interval `[a, a + p)` under the quotient map `𝕜 → AddCircle p` is
the entire space. -/
@[simp]
theorem coe_image_Ico_eq : ((↑) : 𝕜 → AddCircle p) '' Ico a (a + p) = univ := by
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ⊢ Eq (Set.image QuotientAddGroup.mk (Set.Ico a (HAdd.hAdd a p))) Set.univ
  -/
  rw [image_eq_range]
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ⊢ Eq (Set.range fun x => ↑↑x) Set.univ
  -/
  exact (equivIco p a).symm.range_eq_univ
  /-
    🎉 no goals
  -/


/-- The image of the closed-open interval `[a, a + p)` under the quotient map `𝕜 → AddCircle p` is
the entire space. -/
@[simp]
theorem coe_image_Ioc_eq : ((↑) : 𝕜 → AddCircle p) '' Ioc a (a + p) = univ := by
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ⊢ Eq (Set.image QuotientAddGroup.mk (Set.Ioc a (HAdd.hAdd a p))) Set.univ
  -/
  rw [image_eq_range]
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    a : 𝕜
    inst✝ : Archimedean 𝕜
    ⊢ Eq (Set.range fun x => ↑↑x) Set.univ
  -/
  exact (equivIoc p a).symm.range_eq_univ
  /-
    🎉 no goals
  -/


/-- The image of the closed interval `[0, p]` under the quotient map `𝕜 → AddCircle p` is the
entire space. -/
@[simp]
theorem coe_image_Icc_eq : ((↑) : 𝕜 → AddCircle p) '' Icc a (a + p) = univ :=
  eq_top_mono (image_subset _ Ico_subset_Icc_self) <| coe_image_Ico_eq _ _


/-- The rescaling equivalence between additive circles with different periods. -/
def equivAddCircle (hp : p ≠ 0) (hq : q ≠ 0) : AddCircle p ≃+ AddCircle q :=
  QuotientAddGroup.congr _ _ (AddAut.mulRight <| (Units.mk0 p hp)⁻¹ * Units.mk0 q hq) <| by
    rw [AddMonoidHom.map_zmultiples, AddMonoidHom.coe_coe, AddAut.mulRight_apply, Units.val_mul,
      Units.val_mk0, Units.val_inv_eq_inv_val, Units.val_mk0, mul_inv_cancel_left₀ hp]


@[simp]
theorem equivAddCircle_apply_mk (hp : p ≠ 0) (hq : q ≠ 0) (x : 𝕜) :
    equivAddCircle p q hp hq (x : 𝕜) = (x * (p⁻¹ * q) : 𝕜) :=
  rfl


@[simp]
theorem equivAddCircle_symm_apply_mk (hp : p ≠ 0) (hq : q ≠ 0) (x : 𝕜) :
    (equivAddCircle p q hp hq).symm (x : 𝕜) = (x * (q⁻¹ * p) : 𝕜) :=
  rfl


/-- The rescaling homeomorphism between additive circles with different periods. -/
def homeomorphAddCircle (hp : p ≠ 0) (hq : q ≠ 0) : AddCircle p ≃ₜ AddCircle q :=
  ⟨equivAddCircle p q hp hq,
    (continuous_quotient_mk'.comp (continuous_mul_right (p⁻¹ * q))).quotient_lift _,
    (continuous_quotient_mk'.comp (continuous_mul_right (q⁻¹ * p))).quotient_lift _⟩


@[simp]
theorem homeomorphAddCircle_apply_mk (hp : p ≠ 0) (hq : q ≠ 0) (x : 𝕜) :
    homeomorphAddCircle p q hp hq (x : 𝕜) = (x * (p⁻¹ * q) : 𝕜) :=
  rfl


@[simp]
theorem homeomorphAddCircle_symm_apply_mk (hp : p ≠ 0) (hq : q ≠ 0) (x : 𝕜) :
    (homeomorphAddCircle p q hp hq).symm (x : 𝕜) = (x * (q⁻¹ * p) : 𝕜) :=
  rfl

@[simp]
theorem coe_equivIco_mk_apply (x : 𝕜) :
    (equivIco p 0 <| QuotientAddGroup.mk x : 𝕜) = Int.fract (x / p) * p :=
  toIcoMod_eq_fract_mul _ x


instance : DivisibleBy (AddCircle p) ℤ where
  div x n := (↑((n : 𝕜)⁻¹ * (equivIco p 0 x : 𝕜)) : AddCircle p)
  div_zero x := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      p q : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : FloorRing 𝕜
      x : AddCircle p
      ⊢ Eq ((fun x n => ↑(HMul.hMul (Inv.inv ↑n) ↑((AddCircle.equivIco p 0) x))) x 0 …
    -/
    simp only [algebraMap.coe_zero, Int.cast_zero, inv_zero, zero_mul, QuotientAddGroup.mk_zero]
    /-
      🎉 no goals
    -/
  div_cancel {n} x hn := by
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      p q : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : FloorRing 𝕜
      n : Int
      x : AddCircle p
      hn : Ne n 0
      ⊢ Eq (HSMul.hSMul n ((fun x n => ↑(HMul.hMul (Inv.inv ↑n) ↑((AddCircle.equivIc …
    -/
    replace hn : (n : 𝕜) ≠ 0 := by norm_cast
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      p q : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : FloorRing 𝕜
      n : Int
      x : AddCircle p
      hn : Ne (↑n) 0
      ⊢ Eq (HSMul.hSMul n ((fun x n => ↑(HMul.hMul (Inv.inv ↑n) ↑((AddCircle.equivIc …
    -/
    change n • QuotientAddGroup.mk' _ ((n : 𝕜)⁻¹ * ↑(equivIco p 0 x)) = x
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      p q : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : FloorRing 𝕜
      n : Int
      x : AddCircle p
      hn : Ne (↑n) 0
      ⊢ Eq (HSMul.hSMul n ((QuotientAddGroup.mk' (AddSubgroup.zmultiples p)) (HMul.h …
    -/
    rw [← map_zsmul, ← smul_mul_assoc, zsmul_eq_mul, mul_inv_cancel₀ hn, one_mul]
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝¹ : LinearOrderedField 𝕜
      p q : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : FloorRing 𝕜
      n : Int
      x : AddCircle p
      hn : Ne (↑n) 0
      ⊢ Eq ((QuotientAddGroup.mk' (AddSubgroup.zmultiples p)) ↑((AddCircle.equivIco  …
    -/
    exact (equivIco p 0).symm_apply_apply x
    /-
      🎉 no goals
    -/


theorem addOrderOf_period_div {n : ℕ} (h : 0 < n) : addOrderOf ((p / n : 𝕜) : AddCircle p) = n := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    h : LT.lt 0 n
    ⊢ Eq (addOrderOf ↑(HDiv.hDiv p ↑n)) n
  -/
  rw [addOrderOf_eq_iff h]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    h : LT.lt 0 n
    ⊢ And (Eq (HSMul.hSMul n ↑(HDiv.hDiv p ↑n)) 0) (∀ (m : Nat), LT.lt m n → LT.lt …
  -/
  replace h : 0 < (n : 𝕜) := Nat.cast_pos.2 h
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    h : LT.lt 0 ↑n
    ⊢ And (Eq (HSMul.hSMul n ↑(HDiv.hDiv p ↑n)) 0) (∀ (m : Nat), LT.lt m n → LT.lt …
  -/
  refine ⟨?_, fun m hn h0 => ?_⟩ <;> simp only [Ne, ← coe_nsmul, nsmul_eq_mul]
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      h : LT.lt 0 ↑n
      ⊢ Eq (↑(HMul.hMul (↑n) (HDiv.hDiv p ↑n))) 0
    -/
  · rw [mul_div_cancel₀ _ h.ne', coe_period]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    h : LT.lt 0 ↑n
    m : Nat
    hn : LT.lt m n
    h0 : LT.lt 0 m
    ⊢ Not (Eq (↑(HMul.hMul (↑m) (HDiv.hDiv p ↑n))) 0)
  -/
  rw [coe_eq_zero_of_pos_iff p hp.out (mul_pos (Nat.cast_pos.2 h0) <| div_pos hp.out h)]
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    h : LT.lt 0 ↑n
    m : Nat
    hn : LT.lt m n
    h0 : LT.lt 0 m
    ⊢ Not (Exists fun n_1 => Eq (HSMul.hSMul n_1 p) (HMul.hMul (↑m) (HDiv.hDiv p ↑ …
  -/
  rintro ⟨k, hk⟩
  rw [mul_div, eq_div_iff h.ne', nsmul_eq_mul, mul_right_comm, ← Nat.cast_mul,
    (mul_left_injective₀ hp.out.ne').eq_iff, Nat.cast_inj, mul_comm] at hk
  /-
    case refine_2.intro
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    h : LT.lt 0 ↑n
    m : Nat
    hn : LT.lt m n
    h0 : LT.lt 0 m
    k : Nat
    hk : Eq (HMul.hMul n k) m
    ⊢ False
  -/
  exact (Nat.le_of_dvd h0 ⟨_, hk.symm⟩).not_lt hn
  /-
    🎉 no goals
  -/


theorem gcd_mul_addOrderOf_div_eq {n : ℕ} (m : ℕ) (hn : 0 < n) :
    m.gcd n * addOrderOf (↑(↑m / ↑n * p) : AddCircle p) = n := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n m : Nat
    hn : LT.lt 0 n
    ⊢ Eq (HMul.hMul (m.gcd n) (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p))) n
  -/
  rw [mul_comm_div, ← nsmul_eq_mul, coe_nsmul, IsOfFinAddOrder.addOrderOf_nsmul]
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n m : Nat
      hn : LT.lt 0 n
      ⊢ Eq (HMul.hMul (m.gcd n) (HDiv.hDiv (addOrderOf ↑(HDiv.hDiv p ↑n)) ((addOrder …
    -/
  · rw [addOrderOf_period_div hn, Nat.gcd_comm, Nat.mul_div_cancel']
    /-
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n m : Nat
      hn : LT.lt 0 n
      ⊢ Dvd.dvd (n.gcd m) n
    -/
    exact n.gcd_dvd_left m
    /-
      🎉 no goals
    -/
    /-
      case h
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n m : Nat
      hn : LT.lt 0 n
      ⊢ IsOfFinAddOrder ↑(HDiv.hDiv p ↑n)
    -/
  · rwa [← addOrderOf_pos_iff, addOrderOf_period_div hn]
    /-
      🎉 no goals
    -/


theorem addOrderOf_div_of_gcd_eq_one {m n : ℕ} (hn : 0 < n) (h : m.gcd n = 1) :
    addOrderOf (↑(↑m / ↑n * p) : AddCircle p) = n := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    m n : Nat
    hn : LT.lt 0 n
    h : Eq (m.gcd n) 1
    ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p)) n
  -/
  convert gcd_mul_addOrderOf_div_eq p m hn
  /-
    case h.e'_2
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    m n : Nat
    hn : LT.lt 0 n
    h : Eq (m.gcd n) 1
    ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p)) (HMul.hMul (m.gcd n) (addOr …
  -/
  rw [h, one_mul]
  /-
    🎉 no goals
  -/


theorem addOrderOf_div_of_gcd_eq_one' {m : ℤ} {n : ℕ} (hn : 0 < n) (h : m.natAbs.gcd n = 1) :
    addOrderOf (↑(↑m / ↑n * p) : AddCircle p) = n := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    m : Int
    n : Nat
    hn : LT.lt 0 n
    h : Eq (m.natAbs.gcd n) 1
    ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p)) n
  -/
  induction m
    /-
      case ofNat
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      hn : LT.lt 0 n
      a✝ : Nat
      h : Eq ((Int.ofNat a✝).natAbs.gcd n) 1
      ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑(Int.ofNat a✝) ↑n) p)) n
    -/
  · simp only [Int.ofNat_eq_coe, Int.cast_natCast, Int.natAbs_ofNat] at h ⊢
    /-
      case ofNat
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      hn : LT.lt 0 n
      a✝ : Nat
      h : Eq (a✝.gcd n) 1
      ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑a✝ ↑n) p)) n
    -/
    exact addOrderOf_div_of_gcd_eq_one hn h
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      hn : LT.lt 0 n
      a✝ : Nat
      h : Eq ((Int.negSucc a✝).natAbs.gcd n) 1
      ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑(Int.negSucc a✝) ↑n) p)) n
    -/
  · simp only [Int.cast_negSucc, neg_div, neg_mul, coe_neg, addOrderOf_neg]
    /-
      case negSucc
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      hn : LT.lt 0 n
      a✝ : Nat
      h : Eq ((Int.negSucc a✝).natAbs.gcd n) 1
      ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑(HAdd.hAdd a✝ 1) ↑n) p)) n
    -/
    exact addOrderOf_div_of_gcd_eq_one hn h
    /-
      🎉 no goals
    -/


theorem addOrderOf_coe_rat {q : ℚ} : addOrderOf (↑(↑q * p) : AddCircle p) = q.den := by
  have : (↑(q.den : ℤ) : 𝕜) ≠ 0 := by
    norm_cast
    exact q.pos.ne.symm
  rw [← q.num_divInt_den, Rat.cast_divInt_of_ne_zero _ this, Int.cast_natCast, Rat.num_divInt_den,
    addOrderOf_div_of_gcd_eq_one' q.pos q.reduced]


theorem addOrderOf_eq_pos_iff {u : AddCircle p} {n : ℕ} (h : 0 < n) :
    addOrderOf u = n ↔ ∃ m < n, m.gcd n = 1 ∧ ↑(↑m / ↑n * p) = u := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    ⊢ Iff (Eq (addOrderOf u) n) (Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n …
  -/
  refine ⟨QuotientAddGroup.induction_on u fun k hk => ?_, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      u : AddCircle p
      n : Nat
      h : LT.lt 0 n
      ⊢ (Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq (↑(HMul.hMul (HDi …
    -/
  · rintro ⟨m, _, h₁, rfl⟩
    /-
      case refine_1.intro.intro.intro
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      h : LT.lt 0 n
      m : Nat
      left✝ : LT.lt m n
      h₁ : Eq (m.gcd n) 1
      ⊢ Eq (addOrderOf ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p)) n
    -/
    exact addOrderOf_div_of_gcd_eq_one h h₁
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    k : 𝕜
    hk : Eq (addOrderOf ↑k) n
    ⊢ Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq ↑(HMul.hMul (HDiv. …
  -/
  have h0 := addOrderOf_nsmul_eq_zero (k : AddCircle p)
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    k : 𝕜
    hk : Eq (addOrderOf ↑k) n
    h0 : Eq (HSMul.hSMul (addOrderOf ↑k) ↑k) 0
    ⊢ Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq ↑(HMul.hMul (HDiv. …
  -/
  rw [hk, ← coe_nsmul, coe_eq_zero_iff] at h0
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    k : 𝕜
    hk : Eq (addOrderOf ↑k) n
    h0 : Exists fun n_1 => Eq (HSMul.hSMul n_1 p) (HSMul.hSMul n k)
    ⊢ Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq ↑(HMul.hMul (HDiv. …
  -/
  obtain ⟨a, ha⟩ := h0
  /-
    case refine_2.intro
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    k : 𝕜
    hk : Eq (addOrderOf ↑k) n
    a : Int
    ha : Eq (HSMul.hSMul a p) (HSMul.hSMul n k)
    ⊢ Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq ↑(HMul.hMul (HDiv. …
  -/
  have h0 : (_ : 𝕜) ≠ 0 := Nat.cast_ne_zero.2 h.ne'
  rw [nsmul_eq_mul, mul_comm, ← div_eq_iff h0, ← a.ediv_add_emod' n, add_smul, add_div,
    zsmul_eq_mul, Int.cast_mul, Int.cast_natCast, mul_assoc, ← mul_div, mul_comm _ p,
    mul_div_cancel_right₀ p h0] at ha
  /-
    case refine_2.intro
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    k : 𝕜
    hk : Eq (addOrderOf ↑k) n
    a : Int
    ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
    h0 : Ne (↑n) 0
    ⊢ Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq ↑(HMul.hMul (HDiv. …
  -/
  have han : _ = a % n := Int.toNat_of_nonneg (Int.emod_nonneg _ <| mod_cast h.ne')
  have he : (↑(↑((a % n).toNat) / ↑n * p) : AddCircle p) = k := by
    convert congr_arg (QuotientAddGroup.mk : 𝕜 → (AddCircle p)) ha using 1
    rw [coe_add, ← Int.cast_natCast, han, zsmul_eq_mul, mul_div_right_comm, eq_comm,
      add_left_eq_self, ← zsmul_eq_mul, coe_zsmul, coe_period, smul_zero]
  /-
    case refine_2.intro
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    u : AddCircle p
    n : Nat
    h : LT.lt 0 n
    k : 𝕜
    hk : Eq (addOrderOf ↑k) n
    a : Int
    ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
    h0 : Ne (↑n) 0
    han : Eq (↑(HMod.hMod a ↑n).toNat) (HMod.hMod a ↑n)
    he : Eq ↑(HMul.hMul (HDiv.hDiv ↑(HMod.hMod a ↑n).toNat ↑n) p) ↑k
    ⊢ Exists fun m => And (LT.lt m n) (And (Eq (m.gcd n) 1) (Eq ↑(HMul.hMul (HDiv. …
  -/
  refine ⟨(a % n).toNat, ?_, ?_, he⟩
    /-
      case refine_2.intro.refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      u : AddCircle p
      n : Nat
      h : LT.lt 0 n
      k : 𝕜
      hk : Eq (addOrderOf ↑k) n
      a : Int
      ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
      h0 : Ne (↑n) 0
      han : Eq (↑(HMod.hMod a ↑n).toNat) (HMod.hMod a ↑n)
      he : Eq ↑(HMul.hMul (HDiv.hDiv ↑(HMod.hMod a ↑n).toNat ↑n) p) ↑k
      ⊢ LT.lt (HMod.hMod a ↑n).toNat n
    -/
  · rw [← Int.ofNat_lt, han]
    /-
      case refine_2.intro.refine_1
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      u : AddCircle p
      n : Nat
      h : LT.lt 0 n
      k : 𝕜
      hk : Eq (addOrderOf ↑k) n
      a : Int
      ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
      h0 : Ne (↑n) 0
      han : Eq (↑(HMod.hMod a ↑n).toNat) (HMod.hMod a ↑n)
      he : Eq ↑(HMul.hMul (HDiv.hDiv ↑(HMod.hMod a ↑n).toNat ↑n) p) ↑k
      ⊢ LT.lt (HMod.hMod a ↑n) ↑n
    -/
    exact Int.emod_lt_of_pos _ (Int.ofNat_lt.2 h)
    /-
      🎉 no goals
    -/
  · have := (gcd_mul_addOrderOf_div_eq p (Int.toNat (a % ↑n)) h).trans
      ((congr_arg addOrderOf he).trans hk).symm
    /-
      case refine_2.intro.refine_2
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      u : AddCircle p
      n : Nat
      h : LT.lt 0 n
      k : 𝕜
      hk : Eq (addOrderOf ↑k) n
      a : Int
      ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
      h0 : Ne (↑n) 0
      han : Eq (↑(HMod.hMod a ↑n).toNat) (HMod.hMod a ↑n)
      he : Eq ↑(HMul.hMul (HDiv.hDiv ↑(HMod.hMod a ↑n).toNat ↑n) p) ↑k
      this : Eq (HMul.hMul ((HMod.hMod a ↑n).toNat.gcd n) (addOrderOf ↑(HMul.hMul (H …
      ⊢ Eq ((HMod.hMod a ↑n).toNat.gcd n) 1
    -/
    rw [he, Nat.mul_left_eq_self_iff] at this
      /-
        case refine_2.intro.refine_2
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        p : 𝕜
        hp : Fact (LT.lt 0 p)
        u : AddCircle p
        n : Nat
        h : LT.lt 0 n
        k : 𝕜
        hk : Eq (addOrderOf ↑k) n
        a : Int
        ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
        h0 : Ne (↑n) 0
        han : Eq (↑(HMod.hMod a ↑n).toNat) (HMod.hMod a ↑n)
        he : Eq ↑(HMul.hMul (HDiv.hDiv ↑(HMod.hMod a ↑n).toNat ↑n) p) ↑k
        this : Eq ((HMod.hMod a ↑n).toNat.gcd n) 1
        ⊢ Eq ((HMod.hMod a ↑n).toNat.gcd n) 1
      -/
    · exact this
      /-
        🎉 no goals
      -/
      /-
        case refine_2.intro.refine_2
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        p : 𝕜
        hp : Fact (LT.lt 0 p)
        u : AddCircle p
        n : Nat
        h : LT.lt 0 n
        k : 𝕜
        hk : Eq (addOrderOf ↑k) n
        a : Int
        ha : Eq (HAdd.hAdd (HMul.hMul (↑(HDiv.hDiv a ↑n)) p) (HDiv.hDiv (HSMul.hSMul ( …
        h0 : Ne (↑n) 0
        han : Eq (↑(HMod.hMod a ↑n).toNat) (HMod.hMod a ↑n)
        he : Eq ↑(HMul.hMul (HDiv.hDiv ↑(HMod.hMod a ↑n).toNat ↑n) p) ↑k
        this : Eq (HMul.hMul ((HMod.hMod a ↑n).toNat.gcd n) (addOrderOf ↑k)) (addOrder …
        ⊢ LT.lt 0 (addOrderOf ↑k)
      -/
    · rwa [hk]
      /-
        🎉 no goals
      -/


theorem exists_gcd_eq_one_of_isOfFinAddOrder {u : AddCircle p} (h : IsOfFinAddOrder u) :
    ∃ m : ℕ, m.gcd (addOrderOf u) = 1 ∧ m < addOrderOf u ∧ ↑((m : 𝕜) / addOrderOf u * p) = u :=
  let ⟨m, hl, hg, he⟩ := (addOrderOf_eq_pos_iff h.addOrderOf_pos).1 rfl
  ⟨m, hg, hl, he⟩


/-- The natural bijection between points of order `n` and natural numbers less than and coprime to
`n`. The inverse of the map sends `m ↦ (m/n * p : AddCircle p)` where `m` is coprime to `n` and
satisfies `0 ≤ m < n`. -/
def setAddOrderOfEquiv {n : ℕ} (hn : 0 < n) :
    { u : AddCircle p | addOrderOf u = n } ≃ { m | m < n ∧ m.gcd n = 1 } :=
  Equiv.symm <|
    Equiv.ofBijective (fun m => ⟨↑((m : 𝕜) / n * p), addOrderOf_div_of_gcd_eq_one hn m.prop.2⟩)
      (by
        /-
          𝕜 : Type u_1
          B : Type u_2
          inst✝ : LinearOrderedField 𝕜
          p q : 𝕜
          hp : Fact (LT.lt 0 p)
          n : Nat
          hn : LT.lt 0 n
          ⊢ Function.Bijective fun m => ⟨↑(HMul.hMul (HDiv.hDiv ↑↑m ↑n) p), ⋯⟩
        -/
        refine ⟨fun m₁ m₂ h => Subtype.ext ?_, fun u => ?_⟩
          /-
            case refine_1
            𝕜 : Type u_1
            B : Type u_2
            inst✝ : LinearOrderedField 𝕜
            p q : 𝕜
            hp : Fact (LT.lt 0 p)
            n : Nat
            hn : LT.lt 0 n
            m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
            h : Eq ((fun m => ⟨↑(HMul.hMul (HDiv.hDiv ↑↑m ↑n) p), ⋯⟩) m₁) ((fun m => ⟨↑(HM …
            ⊢ Eq ↑m₁ ↑m₂
          -/
        · simp_rw [Subtype.ext_iff] at h
          rw [← sub_eq_zero, ← coe_sub, ← sub_mul, ← sub_div, ← Int.cast_natCast m₁,
            ← Int.cast_natCast m₂, ← Int.cast_sub, coe_eq_zero_iff] at h
          /-
            case refine_1
            𝕜 : Type u_1
            B : Type u_2
            inst✝ : LinearOrderedField 𝕜
            p q : 𝕜
            hp : Fact (LT.lt 0 p)
            n : Nat
            hn : LT.lt 0 n
            m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
            h : Exists fun n_1 => Eq (HSMul.hSMul n_1 p) (HMul.hMul (HDiv.hDiv ↑(HSub.hSub …
            ⊢ Eq ↑m₁ ↑m₂
          -/
          obtain ⟨m, hm⟩ := h
          rw [← mul_div_right_comm, eq_div_iff, mul_comm, ← zsmul_eq_mul, mul_smul_comm, ←
            nsmul_eq_mul, ← natCast_zsmul, smul_smul,
            zsmul_left_inj hp.out, mul_comm] at hm
          /-
            case refine_1.intro
            𝕜 : Type u_1
            B : Type u_2
            inst✝ : LinearOrderedField 𝕜
            p q : 𝕜
            hp : Fact (LT.lt 0 p)
            n : Nat
            hn : LT.lt 0 n
            m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
            m : Int
            hm : Eq (HMul.hMul (↑n) m) (HSub.hSub ↑↑m₁ ↑↑m₂)
            ⊢ Eq ↑m₁ ↑m₂
          -/
          swap
            /-
              case refine_1.intro
              𝕜 : Type u_1
              B : Type u_2
              inst✝ : LinearOrderedField 𝕜
              p q : 𝕜
              hp : Fact (LT.lt 0 p)
              n : Nat
              hn : LT.lt 0 n
              m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
              m : Int
              hm : Eq (HSMul.hSMul m p) (HDiv.hDiv (HMul.hMul (↑(HSub.hSub ↑↑m₁ ↑↑m₂)) p) ↑n)
              ⊢ Ne (↑n) 0
            -/
          · exact Nat.cast_ne_zero.2 hn.ne'
            /-
              🎉 no goals
            -/
          /-
            case refine_1.intro
            𝕜 : Type u_1
            B : Type u_2
            inst✝ : LinearOrderedField 𝕜
            p q : 𝕜
            hp : Fact (LT.lt 0 p)
            n : Nat
            hn : LT.lt 0 n
            m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
            m : Int
            hm : Eq (HMul.hMul (↑n) m) (HSub.hSub ↑↑m₁ ↑↑m₂)
            ⊢ Eq ↑m₁ ↑m₂
          -/
          rw [← @Nat.cast_inj ℤ, ← sub_eq_zero]
          /-
            case refine_1.intro
            𝕜 : Type u_1
            B : Type u_2
            inst✝ : LinearOrderedField 𝕜
            p q : 𝕜
            hp : Fact (LT.lt 0 p)
            n : Nat
            hn : LT.lt 0 n
            m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
            m : Int
            hm : Eq (HMul.hMul (↑n) m) (HSub.hSub ↑↑m₁ ↑↑m₂)
            ⊢ Eq (HSub.hSub ↑↑m₁ ↑↑m₂) 0
          -/
          refine Int.eq_zero_of_abs_lt_dvd ⟨_, hm.symm⟩ (abs_sub_lt_iff.2 ⟨?_, ?_⟩) <;>
            /-
              case refine_1.intro.refine_1
              𝕜 : Type u_1
              B : Type u_2
              inst✝ : LinearOrderedField 𝕜
              p q : 𝕜
              hp : Fact (LT.lt 0 p)
              n : Nat
              hn : LT.lt 0 n
              m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
              m : Int
              hm : Eq (HMul.hMul (↑n) m) (HSub.hSub ↑↑m₁ ↑↑m₂)
              ⊢ LT.lt (HSub.hSub ↑↑m₁ ↑↑m₂) ↑n
            -/
            apply (Int.sub_le_self _ <| Nat.cast_nonneg _).trans_lt (Nat.cast_lt.2 _)
          /-
            𝕜 : Type u_1
            B : Type u_2
            inst✝ : LinearOrderedField 𝕜
            p q : 𝕜
            hp : Fact (LT.lt 0 p)
            n : Nat
            hn : LT.lt 0 n
            m₁ m₂ : ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))
            m : Int
            hm : Eq (HMul.hMul (↑n) m) (HSub.hSub ↑↑m₁ ↑↑m₂)
            ⊢ LT.lt (↑m₁) n
          -/
          exacts [m₁.2.1, m₂.2.1]
          /-
            🎉 no goals
          -/
        /-
          case refine_2
          𝕜 : Type u_1
          B : Type u_2
          inst✝ : LinearOrderedField 𝕜
          p q : 𝕜
          hp : Fact (LT.lt 0 p)
          n : Nat
          hn : LT.lt 0 n
          u : ↑(setOf fun u => Eq (addOrderOf u) n)
          ⊢ Exists fun a => Eq ((fun m => ⟨↑(HMul.hMul (HDiv.hDiv ↑↑m ↑n) p), ⋯⟩) a) u
        -/
        obtain ⟨m, hmn, hg, he⟩ := (addOrderOf_eq_pos_iff hn).mp u.2
        /-
          case refine_2.intro.intro.intro
          𝕜 : Type u_1
          B : Type u_2
          inst✝ : LinearOrderedField 𝕜
          p q : 𝕜
          hp : Fact (LT.lt 0 p)
          n : Nat
          hn : LT.lt 0 n
          u : ↑(setOf fun u => Eq (addOrderOf u) n)
          m : Nat
          hmn : LT.lt m n
          hg : Eq (m.gcd n) 1
          he : Eq ↑(HMul.hMul (HDiv.hDiv ↑m ↑n) p) ↑u
          ⊢ Exists fun a => Eq ((fun m => ⟨↑(HMul.hMul (HDiv.hDiv ↑↑m ↑n) p), ⋯⟩) a) u
        -/
        exact ⟨⟨m, hmn, hg⟩, Subtype.ext he⟩)
        /-
          🎉 no goals
        -/


@[simp]
theorem card_addOrderOf_eq_totient {n : ℕ} :
    Nat.card { u : AddCircle p // addOrderOf u = n } = n.totient := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p : 𝕜
    hp : Fact (LT.lt 0 p)
    n : Nat
    ⊢ Eq (Nat.card (Subtype fun u => Eq (addOrderOf u) n)) n.totient
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      ⊢ Eq (Nat.card (Subtype fun u => Eq (addOrderOf u) 0)) (Nat.totient 0)
    -/
  · simp only [Nat.totient_zero, addOrderOf_eq_zero_iff]
    /-
      case inl
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      ⊢ Eq (Nat.card (Subtype fun u => Not (IsOfFinAddOrder u))) 0
    -/
    rcases em (∃ u : AddCircle p, ¬IsOfFinAddOrder u) with (⟨u, hu⟩ | h)
    · have : Infinite { u : AddCircle p // ¬IsOfFinAddOrder u } := by
        erw [infinite_coe_iff]
        exact infinite_not_isOfFinAddOrder hu
      /-
        case inl.inl.intro
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        p : 𝕜
        hp : Fact (LT.lt 0 p)
        u : AddCircle p
        hu : Not (IsOfFinAddOrder u)
        this : Infinite (Subtype fun u => Not (IsOfFinAddOrder u))
        ⊢ Eq (Nat.card (Subtype fun u => Not (IsOfFinAddOrder u))) 0
      -/
      exact Nat.card_eq_zero_of_infinite
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        p : 𝕜
        hp : Fact (LT.lt 0 p)
        h : Not (Exists fun u => Not (IsOfFinAddOrder u))
        ⊢ Eq (Nat.card (Subtype fun u => Not (IsOfFinAddOrder u))) 0
      -/
    · have : IsEmpty { u : AddCircle p // ¬IsOfFinAddOrder u } := by simpa [isEmpty_subtype] using h
      /-
        case inl.inr
        𝕜 : Type u_1
        inst✝ : LinearOrderedField 𝕜
        p : 𝕜
        hp : Fact (LT.lt 0 p)
        h : Not (Exists fun u => Not (IsOfFinAddOrder u))
        this : IsEmpty (Subtype fun u => Not (IsOfFinAddOrder u))
        ⊢ Eq (Nat.card (Subtype fun u => Not (IsOfFinAddOrder u))) 0
      -/
      exact Nat.card_of_isEmpty
      /-
        🎉 no goals
      -/
  · rw [← coe_setOf, Nat.card_congr (setAddOrderOfEquiv p hn),
      n.totient_eq_card_lt_and_coprime]
    /-
      case inr
      𝕜 : Type u_1
      inst✝ : LinearOrderedField 𝕜
      p : 𝕜
      hp : Fact (LT.lt 0 p)
      n : Nat
      hn : GT.gt n 0
      ⊢ Eq (Nat.card ↑(setOf fun m => And (LT.lt m n) (Eq (m.gcd n) 1))) (Nat.card ↑ …
    -/
    simp only [Nat.gcd_comm]
    /-
      🎉 no goals
    -/


theorem finite_setOf_add_order_eq {n : ℕ} (hn : 0 < n) :
    { u : AddCircle p | addOrderOf u = n }.Finite :=
                                                        /-
                                                          𝕜 : Type u_1
                                                          inst✝ : LinearOrderedField 𝕜
                                                          p : 𝕜
                                                          hp : Fact (LT.lt 0 p)
                                                          n : Nat
                                                          hn : LT.lt 0 n
                                                          ⊢ Ne (Nat.card ↑(setOf fun u => Eq (addOrderOf u) n)) 0
                                                        -/
  finite_coe_iff.mp <| Nat.finite_of_card_ne_zero <| by simp [hn.ne']
                                                        /-
                                                          🎉 no goals
                                                        -/


instance pathConnectedSpace : PathConnectedSpace <| AddCircle p :=
  (inferInstance : PathConnectedSpace (Quotient _))


/-- The "additive circle" `ℝ ⧸ (ℤ ∙ p)` is compact. -/
instance compactSpace [Fact (0 < p)] : CompactSpace <| AddCircle p := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    p : Real
    inst✝ : Fact (LT.lt 0 p)
    ⊢ CompactSpace (AddCircle p)
  -/
  rw [← isCompact_univ_iff, ← coe_image_Icc_eq p 0]
  /-
    𝕜 : Type u_1
    B : Type u_2
    p : Real
    inst✝ : Fact (LT.lt 0 p)
    ⊢ IsCompact (Set.image QuotientAddGroup.mk (Set.Icc 0 (HAdd.hAdd 0 p)))
  -/
  exact isCompact_Icc.image (AddCircle.continuous_mk' p)
  /-
    🎉 no goals
  -/


/-- The action on `ℝ` by right multiplication of its the subgroup `zmultiples p` (the multiples of
`p:ℝ`) is properly discontinuous. -/
instance : ProperlyDiscontinuousVAdd (zmultiples p).op ℝ :=
  (zmultiples p).properlyDiscontinuousVAdd_opposite_of_tendsto_cofinite
    (AddSubgroup.tendsto_zmultiples_subtype_cofinite p)


instance instZeroLTOne [StrictOrderedSemiring 𝕜] : Fact ((0 : 𝕜) < 1) := ⟨zero_lt_one⟩


/-- The unit circle `ℝ ⧸ ℤ`. -/
abbrev UnitAddCircle :=
  AddCircle (1 : ℝ)


local notation "𝕋" => AddCircle p


/-- The relation identifying the endpoints of `Icc a (a + p)`. -/
inductive EndpointIdent : Icc a (a + p) → Icc a (a + p) → Prop
  | mk :
    EndpointIdent ⟨a, left_mem_Icc.mpr <| le_add_of_nonneg_right hp.out.le⟩
      ⟨a + p, right_mem_Icc.mpr <| le_add_of_nonneg_right hp.out.le⟩


/-- The equivalence between `AddCircle p` and the quotient of `[a, a + p]` by the relation
identifying the endpoints. -/
def equivIccQuot : 𝕋 ≃ Quot (EndpointIdent p a) where
  toFun x := Quot.mk _ <| inclusion Ico_subset_Icc_self (equivIco _ _ x)
  invFun x :=
    Quot.liftOn x (↑) <| by
      /-
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹ : LinearOrderedAddCommGroup 𝕜
        p a : 𝕜
        hp : Fact (LT.lt 0 p)
        inst✝ : Archimedean 𝕜
        x : Quot (AddCircle.EndpointIdent p a)
        ⊢ ∀ (a_1 b : ↑(Set.Icc a (HAdd.hAdd a p))), AddCircle.EndpointIdent p a a_1 b  …
      -/
      rintro _ _ ⟨_⟩
      /-
        case mk
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹ : LinearOrderedAddCommGroup 𝕜
        p a : 𝕜
        hp : Fact (LT.lt 0 p)
        inst✝ : Archimedean 𝕜
        x : Quot (AddCircle.EndpointIdent p a)
        ⊢ Eq ((fun x => ↑↑x) ⟨a, ⋯⟩) ((fun x => ↑↑x) ⟨HAdd.hAdd a p, ⋯⟩)
      -/
      exact (coe_add_period p a).symm
      /-
        🎉 no goals
      -/
  left_inv := (equivIco p a).symm_apply_apply
  right_inv :=
    Quot.ind <| by
      /-
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹ : LinearOrderedAddCommGroup 𝕜
        p a : 𝕜
        hp : Fact (LT.lt 0 p)
        inst✝ : Archimedean 𝕜
        ⊢ ∀ (a_1 : ↑(Set.Icc a (HAdd.hAdd a p))), Eq ((fun x => Quot.mk (AddCircle.End …
      -/
      rintro ⟨x, hx⟩
      /-
        case mk
        𝕜 : Type u_1
        B : Type u_2
        inst✝¹ : LinearOrderedAddCommGroup 𝕜
        p a : 𝕜
        hp : Fact (LT.lt 0 p)
        inst✝ : Archimedean 𝕜
        x : 𝕜
        hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x
        ⊢ Eq ((fun x => Quot.mk (AddCircle.EndpointIdent p a) (Set.inclusion ⋯ ((AddCi …
      -/
      rcases ne_or_eq x (a + p) with (h | rfl)
        /-
          case mk.inl
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          x : 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x
          h : Ne x (HAdd.hAdd a p)
          ⊢ Eq ((fun x => Quot.mk (AddCircle.EndpointIdent p a) (Set.inclusion ⋯ ((AddCi …
        -/
      · revert x
        /-
          case mk.inl
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          ⊢ ∀ (x : 𝕜) (hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x), Ne x (HAdd.hA …
        -/
        dsimp only
        /-
          case mk.inl
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          ⊢ ∀ (x : 𝕜) (hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x), Ne x (HAdd.hA …
        -/
        intro x hx h
        /-
          case mk.inl
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          x : 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x
          h : Ne x (HAdd.hAdd a p)
          ⊢ Eq (Quot.mk (AddCircle.EndpointIdent p a) (Set.inclusion ⋯ ((AddCircle.equiv …
        -/
        congr
        /-
          case mk.inl.e_a
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          x : 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x
          h : Ne x (HAdd.hAdd a p)
          ⊢ Eq (Set.inclusion ⋯ ((AddCircle.equivIco p a) ((Quot.mk (AddCircle.EndpointI …
        -/
        ext1
        /-
          case mk.inl.e_a.a
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          x : 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) x
          h : Ne x (HAdd.hAdd a p)
          ⊢ Eq ↑(Set.inclusion ⋯ ((AddCircle.equivIco p a) ((Quot.mk (AddCircle.Endpoint …
        -/
        apply congr_arg Subtype.val ((equivIco p a).right_inv ⟨x, hx.1, hx.2.lt_of_ne h⟩)
        /-
          🎉 no goals
        -/
        /-
          case mk.inr
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) (HAdd.hAdd a p)
          ⊢ Eq ((fun x => Quot.mk (AddCircle.EndpointIdent p a) (Set.inclusion ⋯ ((AddCi …
        -/
      · rw [← Quot.sound EndpointIdent.mk]
        /-
          case mk.inr
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) (HAdd.hAdd a p)
          ⊢ Eq ((fun x => Quot.mk (AddCircle.EndpointIdent p a) (Set.inclusion ⋯ ((AddCi …
        -/
        dsimp only
        /-
          case mk.inr
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) (HAdd.hAdd a p)
          ⊢ Eq (Quot.mk (AddCircle.EndpointIdent p a) (Set.inclusion ⋯ ((AddCircle.equiv …
        -/
        congr
        /-
          case mk.inr.e_a
          𝕜 : Type u_1
          B : Type u_2
          inst✝¹ : LinearOrderedAddCommGroup 𝕜
          p a : 𝕜
          hp : Fact (LT.lt 0 p)
          inst✝ : Archimedean 𝕜
          hx : Membership.mem (Set.Icc a (HAdd.hAdd a p)) (HAdd.hAdd a p)
          ⊢ Eq (Set.inclusion ⋯ ((AddCircle.equivIco p a) ((Quot.mk (AddCircle.EndpointI …
        -/
        ext1
        apply congr_arg Subtype.val
          ((equivIco p a).right_inv ⟨a, le_refl a, lt_add_of_pos_right a hp.out⟩)


theorem equivIccQuot_comp_mk_eq_toIcoMod :
    equivIccQuot p a ∘ Quotient.mk'' = fun x =>
      Quot.mk _ ⟨toIcoMod hp.out a x, Ico_subset_Icc_self <| toIcoMod_mem_Ico _ _ x⟩ :=
  rfl


theorem equivIccQuot_comp_mk_eq_toIocMod :
    equivIccQuot p a ∘ Quotient.mk'' = fun x =>
      Quot.mk _ ⟨toIocMod hp.out a x, Ioc_subset_Icc_self <| toIocMod_mem_Ioc _ _ x⟩ := by
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p a : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝ : Archimedean 𝕜
    ⊢ Eq (Function.comp (⇑(AddCircle.equivIccQuot p a)) Quotient.mk'') fun x => Qu …
  -/
  rw [equivIccQuot_comp_mk_eq_toIcoMod]
  /-
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p a : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝ : Archimedean 𝕜
    ⊢ Eq (fun x => Quot.mk (AddCircle.EndpointIdent p a) ⟨toIcoMod ⋯ a x, ⋯⟩) fun  …
  -/
  funext x
  /-
    case h
    𝕜 : Type u_1
    inst✝¹ : LinearOrderedAddCommGroup 𝕜
    p a : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝ : Archimedean 𝕜
    x : 𝕜
    ⊢ Eq (Quot.mk (AddCircle.EndpointIdent p a) ⟨toIcoMod ⋯ a x, ⋯⟩) (Quot.mk (Add …
  -/
  by_cases h : a ≡ x [PMOD p]
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : LinearOrderedAddCommGroup 𝕜
      p a : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : Archimedean 𝕜
      x : 𝕜
      h : AddCommGroup.ModEq p a x
      ⊢ Eq (Quot.mk (AddCircle.EndpointIdent p a) ⟨toIcoMod ⋯ a x, ⋯⟩) (Quot.mk (Add …
    -/
  · simp_rw [(modEq_iff_toIcoMod_eq_left hp.out).1 h, (modEq_iff_toIocMod_eq_right hp.out).1 h]
    /-
      case pos
      𝕜 : Type u_1
      inst✝¹ : LinearOrderedAddCommGroup 𝕜
      p a : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : Archimedean 𝕜
      x : 𝕜
      h : AddCommGroup.ModEq p a x
      ⊢ Eq (Quot.mk (AddCircle.EndpointIdent p a) ⟨a, ⋯⟩) (Quot.mk (AddCircle.Endpoi …
    -/
    exact Quot.sound EndpointIdent.mk
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝¹ : LinearOrderedAddCommGroup 𝕜
      p a : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝ : Archimedean 𝕜
      x : 𝕜
      h : Not (AddCommGroup.ModEq p a x)
      ⊢ Eq (Quot.mk (AddCircle.EndpointIdent p a) ⟨toIcoMod ⋯ a x, ⋯⟩) (Quot.mk (Add …
    -/
  · simp_rw [(not_modEq_iff_toIcoMod_eq_toIocMod hp.out).1 h]
    /-
      🎉 no goals
    -/


/-- The natural map from `[a, a + p] ⊂ 𝕜` with endpoints identified to `𝕜 / ℤ • p`, as a
homeomorphism of topological spaces. -/
def homeoIccQuot [TopologicalSpace 𝕜] [OrderTopology 𝕜] : 𝕋 ≃ₜ Quot (EndpointIdent p a) where
  toEquiv := equivIccQuot p a
  continuous_toFun := by
    simp_rw [isQuotientMap_quotient_mk'.continuous_iff, continuous_iff_continuousAt,
      continuousAt_iff_continuous_left_right]
    /-
      𝕜 : Type u_1
      B : Type u_2
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      p a : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      ⊢ ∀ (x : 𝕜), And (ContinuousWithinAt (Function.comp (AddCircle.equivIccQuot p  …
    -/
    intro x; constructor
    /-
      case left
      𝕜 : Type u_1
      B : Type u_2
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      p a : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      x : 𝕜
      ⊢ ContinuousWithinAt (Function.comp (AddCircle.equivIccQuot p a).toFun Quotien …
    -/
    on_goal 1 => erw [equivIccQuot_comp_mk_eq_toIocMod]
    /-
      case left
      𝕜 : Type u_1
      B : Type u_2
      inst✝³ : LinearOrderedAddCommGroup 𝕜
      p a : 𝕜
      hp : Fact (LT.lt 0 p)
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      x : 𝕜
      ⊢ ContinuousWithinAt (fun x => Quot.mk (AddCircle.EndpointIdent p a) ⟨toIocMod …
    -/
    on_goal 2 => erw [equivIccQuot_comp_mk_eq_toIcoMod]
    all_goals
      apply continuous_quot_mk.continuousAt.comp_continuousWithinAt
      rw [IsInducing.subtypeVal.continuousWithinAt_iff]
      /-
        case left
        𝕜 : Type u_1
        B : Type u_2
        inst✝³ : LinearOrderedAddCommGroup 𝕜
        p a : 𝕜
        hp : Fact (LT.lt 0 p)
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        x : 𝕜
        ⊢ ContinuousWithinAt (Function.comp Subtype.val fun x => ⟨toIocMod ⋯ a x, ⋯⟩)  …
      -/
    · apply continuous_left_toIocMod
      /-
        🎉 no goals
      -/
      /-
        case right
        𝕜 : Type u_1
        B : Type u_2
        inst✝³ : LinearOrderedAddCommGroup 𝕜
        p a : 𝕜
        hp : Fact (LT.lt 0 p)
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        x : 𝕜
        ⊢ ContinuousWithinAt (Function.comp Subtype.val fun x => ⟨toIcoMod ⋯ a x, ⋯⟩)  …
      -/
    · apply continuous_right_toIcoMod
      /-
        🎉 no goals
      -/
  continuous_invFun :=
    continuous_quot_lift _ ((AddCircle.continuous_mk' p).comp continuous_subtype_val)


theorem liftIco_eq_lift_Icc {f : 𝕜 → B} (h : f a = f (a + p)) :
    liftIco p a f =
      Quot.lift (restrict (Icc a <| a + p) f)
          (by
            /-
              𝕜 : Type u_1
              B : Type u_2
              inst✝¹ : LinearOrderedAddCommGroup 𝕜
              p a : 𝕜
              hp : Fact (LT.lt 0 p)
              inst✝ : Archimedean 𝕜
              f : 𝕜 → B
              h : Eq (f a) (f (HAdd.hAdd a p))
              ⊢ ∀ (a_1 b : ↑(Set.Icc a (HAdd.hAdd a p))), AddCircle.EndpointIdent p a a_1 b  …
            -/
            rintro _ _ ⟨_⟩
            /-
              case mk
              𝕜 : Type u_1
              B : Type u_2
              inst✝¹ : LinearOrderedAddCommGroup 𝕜
              p a : 𝕜
              hp : Fact (LT.lt 0 p)
              inst✝ : Archimedean 𝕜
              f : 𝕜 → B
              h : Eq (f a) (f (HAdd.hAdd a p))
              ⊢ Eq ((Set.Icc a (HAdd.hAdd a p)).restrict f ⟨a, ⋯⟩) ((Set.Icc a (HAdd.hAdd a  …
            -/
            exact h) ∘
            /-
              🎉 no goals
            -/
        equivIccQuot p a :=
  rfl


theorem liftIco_zero_coe_apply {f : 𝕜 → B} {x : 𝕜} (hx : x ∈ Ico 0 p) : liftIco p 0 f ↑x = f x :=
                        /-
                          𝕜 : Type u_1
                          B : Type u_2
                          inst✝¹ : LinearOrderedAddCommGroup 𝕜
                          p : 𝕜
                          hp : Fact (LT.lt 0 p)
                          inst✝ : Archimedean 𝕜
                          f : 𝕜 → B
                          x : 𝕜
                          hx : Membership.mem (Set.Ico 0 p) x
                          ⊢ Membership.mem (Set.Ico 0 (HAdd.hAdd 0 p)) x
                        -/
  liftIco_coe_apply (by rwa [zero_add])
                        /-
                          🎉 no goals
                        -/


theorem liftIco_continuous [TopologicalSpace B] {f : 𝕜 → B} (hf : f a = f (a + p))
    (hc : ContinuousOn f <| Icc a (a + p)) : Continuous (liftIco p a f) := by
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝⁴ : LinearOrderedAddCommGroup 𝕜
    p a : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝³ : Archimedean 𝕜
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : TopologicalSpace B
    f : 𝕜 → B
    hf : Eq (f a) (f (HAdd.hAdd a p))
    hc : ContinuousOn f (Set.Icc a (HAdd.hAdd a p))
    ⊢ Continuous (AddCircle.liftIco p a f)
  -/
  rw [liftIco_eq_lift_Icc hf]
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝⁴ : LinearOrderedAddCommGroup 𝕜
    p a : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝³ : Archimedean 𝕜
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : TopologicalSpace B
    f : 𝕜 → B
    hf : Eq (f a) (f (HAdd.hAdd a p))
    hc : ContinuousOn f (Set.Icc a (HAdd.hAdd a p))
    ⊢ Continuous (Function.comp (Quot.lift ((Set.Icc a (HAdd.hAdd a p)).restrict f …
  -/
  refine Continuous.comp ?_ (homeoIccQuot p a).continuous_toFun
  /-
    𝕜 : Type u_1
    B : Type u_2
    inst✝⁴ : LinearOrderedAddCommGroup 𝕜
    p a : 𝕜
    hp : Fact (LT.lt 0 p)
    inst✝³ : Archimedean 𝕜
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : TopologicalSpace B
    f : 𝕜 → B
    hf : Eq (f a) (f (HAdd.hAdd a p))
    hc : ContinuousOn f (Set.Icc a (HAdd.hAdd a p))
    ⊢ Continuous (Quot.lift ((Set.Icc a (HAdd.hAdd a p)).restrict f) ⋯)
  -/
  exact continuous_coinduced_dom.mpr (continuousOn_iff_continuous_restrict.mp hc)
  /-
    🎉 no goals
  -/


theorem liftIco_zero_continuous [TopologicalSpace B] {f : 𝕜 → B} (hf : f 0 = f p)
    (hc : ContinuousOn f <| Icc 0 p) : Continuous (liftIco p 0 f) :=
                         /-
                           𝕜 : Type u_1
                           B : Type u_2
                           inst✝⁴ : LinearOrderedAddCommGroup 𝕜
                           p : 𝕜
                           hp : Fact (LT.lt 0 p)
                           inst✝³ : Archimedean 𝕜
                           inst✝² : TopologicalSpace 𝕜
                           inst✝¹ : OrderTopology 𝕜
                           inst✝ : TopologicalSpace B
                           f : 𝕜 → B
                           hf : Eq (f 0) (f p)
                           hc : ContinuousOn f (Set.Icc 0 p)
                           ⊢ Eq (f 0) (f (HAdd.hAdd 0 p))
                         -/
                         /-
                           🎉 no goals
                         -/
  liftIco_continuous (by rwa [zero_add] : f 0 = f (0 + p)) (by rwa [zero_add])
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The `AddMonoidHom` from `ZMod N` to `ℝ / ℤ` sending `j mod N` to `j / N mod 1`. -/
noncomputable def toAddCircle : ZMod N →+ UnitAddCircle :=
                                                      /-
                                                        𝕜 : Type u_1
                                                        B : Type u_2
                                                        N : Nat
                                                        inst✝ : NeZero N
                                                        ⊢ ∀ (a b : Int), Eq ((fun j => ↑(HDiv.hDiv ↑j ↑N)) (HAdd.hAdd a b)) (HAdd.hAdd …
                                                      -/
  lift N ⟨AddMonoidHom.mk' (fun j ↦ ↑(j / N : ℝ)) (by simp [add_div]),
                                                      /-
                                                        🎉 no goals
                                                      -/
       /-
         𝕜 : Type u_1
         B : Type u_2
         N : Nat
         inst✝ : NeZero N
         ⊢ Eq ((AddMonoidHom.mk' (fun j => ↑(HDiv.hDiv ↑j ↑N)) ⋯) ↑N) 0
       -/
    by simp [div_self (NeZero.ne _)]⟩
       /-
         🎉 no goals
       -/


lemma toAddCircle_intCast (j : ℤ) :
    toAddCircle (j : ZMod N) = ↑(j / N : ℝ) := by
  /-
    N : Nat
    inst✝ : NeZero N
    j : Int
    ⊢ Eq (ZMod.toAddCircle ↑j) ↑(HDiv.hDiv ↑j ↑N)
  -/
  simp [toAddCircle]
  /-
    🎉 no goals
  -/


lemma toAddCircle_natCast (j : ℕ) :
    toAddCircle (j : ZMod N) = ↑(j / N : ℝ) := by
  /-
    N : Nat
    inst✝ : NeZero N
    j : Nat
    ⊢ Eq (ZMod.toAddCircle ↑j) ↑(HDiv.hDiv ↑j ↑N)
  -/
  simpa using toAddCircle_intCast (N := N) j
  /-
    🎉 no goals
  -/


/--
Explicit formula for `toCircle j`. Note that this is "evil" because it uses `ZMod.val`. Where
possible, it is recommended to lift `j` to `ℤ` and use `toAddCircle_intCast` instead.
-/
lemma toAddCircle_apply (j : ZMod N) :
    toAddCircle j = ↑(j.val / N : ℝ) := by
  /-
    N : Nat
    inst✝ : NeZero N
    j : ZMod N
    ⊢ Eq (ZMod.toAddCircle j) ↑(HDiv.hDiv ↑j.val ↑N)
  -/
  rw [← toAddCircle_natCast, natCast_zmod_val]
  /-
    🎉 no goals
  -/


variable (N) in
lemma toAddCircle_injective : Function.Injective (toAddCircle : ZMod N → _) := by
  /-
    N : Nat
    inst✝ : NeZero N
    ⊢ Function.Injective ⇑ZMod.toAddCircle
  -/
  intro x y hxy
  /-
    N : Nat
    inst✝ : NeZero N
    x y : ZMod N
    hxy : Eq (ZMod.toAddCircle x) (ZMod.toAddCircle y)
    ⊢ Eq x y
  -/
  have : (0 : ℝ) < N := Nat.cast_pos.mpr (NeZero.pos _)
  rwa [toAddCircle_apply, toAddCircle_apply, AddCircle.coe_eq_coe_iff_of_mem_Ico
    (hp := Real.fact_zero_lt_one) (a := 0), div_left_inj' this.ne', Nat.cast_inj,
    (val_injective N).eq_iff] at hxy <;>
  /-
    case hx
    N : Nat
    inst✝ : NeZero N
    x y : ZMod N
    hxy : Eq ↑(HDiv.hDiv ↑x.val ↑N) ↑(HDiv.hDiv ↑y.val ↑N)
    this : LT.lt 0 ↑N
    ⊢ Membership.mem (Set.Ico 0 (HAdd.hAdd 0 1)) (HDiv.hDiv ↑x.val ↑N)
  -/
  /-
    🎉 no goals
  -/
  exact ⟨by positivity, by simpa only [zero_add, div_lt_one this, Nat.cast_lt] using val_lt _⟩
  /-
    🎉 no goals
  -/


@[simp] lemma toAddCircle_inj {j k : ZMod N} : toAddCircle j = toAddCircle k ↔ j = k :=
  (toAddCircle_injective N).eq_iff


@[simp] lemma toAddCircle_eq_zero {j : ZMod N} : toAddCircle j = 0 ↔ j = 0 :=
  map_eq_zero_iff _ (toAddCircle_injective N)


