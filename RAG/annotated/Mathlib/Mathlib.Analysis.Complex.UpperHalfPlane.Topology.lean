instance : TopologicalSpace ℍ :=
  instTopologicalSpaceSubtype


theorem isOpenEmbedding_coe : IsOpenEmbedding ((↑) : ℍ → ℂ) :=
  IsOpen.isOpenEmbedding_subtypeVal <| isOpen_lt continuous_const Complex.continuous_im


@[deprecated (since := "2024-10-18")]
alias openEmbedding_coe := isOpenEmbedding_coe


theorem isEmbedding_coe : IsEmbedding ((↑) : ℍ → ℂ) :=
  IsEmbedding.subtypeVal


@[deprecated (since := "2024-10-26")]
alias embedding_coe := isEmbedding_coe


theorem continuous_coe : Continuous ((↑) : ℍ → ℂ) :=
  isEmbedding_coe.continuous


theorem continuous_re : Continuous re :=
  Complex.continuous_re.comp continuous_coe


theorem continuous_im : Continuous im :=
  Complex.continuous_im.comp continuous_coe


instance : SecondCountableTopology ℍ :=
  TopologicalSpace.Subtype.secondCountableTopology _


instance : T3Space ℍ := Subtype.t3Space


instance : T4Space ℍ := inferInstance


instance : ContractibleSpace ℍ :=
  (convex_halfSpace_im_gt 0).contractibleSpace ⟨I, one_pos.trans_eq I_im.symm⟩


instance : LocPathConnectedSpace ℍ := isOpenEmbedding_coe.locPathConnectedSpace


instance : NoncompactSpace ℍ := by
  /-
    ⊢ NoncompactSpace UpperHalfPlane
  -/
  refine ⟨fun h => ?_⟩
  /-
    h : IsCompact Set.univ
    ⊢ False
  -/
  have : IsCompact (Complex.im ⁻¹' Ioi 0) := isCompact_iff_isCompact_univ.2 h
  /-
    h : IsCompact Set.univ
    this : IsCompact (Set.preimage Complex.im (Set.Ioi 0))
    ⊢ False
  -/
  replace := this.isClosed.closure_eq
  /-
    h : IsCompact Set.univ
    this : Eq (closure (Set.preimage Complex.im (Set.Ioi 0))) (Set.preimage Comple …
    ⊢ False
  -/
  rw [closure_preimage_im, closure_Ioi, Set.ext_iff] at this
  /-
    h : IsCompact Set.univ
    this : ∀ (x : Complex), Iff (Membership.mem (Set.preimage Complex.im (Set.Ici  …
    ⊢ False
  -/
  exact absurd ((this 0).1 (@left_mem_Ici ℝ _ 0)) (@lt_irrefl ℝ _ 0)
  /-
    🎉 no goals
  -/


instance : LocallyCompactSpace ℍ :=
  isOpenEmbedding_coe.locallyCompactSpace


/-- The vertical strip of width `A` and height `B`, defined by elements whose real part has absolute
value less than or equal to `A` and imaginary part is at least `B`. -/
def verticalStrip (A B : ℝ) := {z : ℍ | |z.re| ≤ A ∧ B ≤ z.im}


theorem mem_verticalStrip_iff (A B : ℝ) (z : ℍ) : z ∈ verticalStrip A B ↔ |z.re| ≤ A ∧ B ≤ z.im :=
  Iff.rfl


@[gcongr]
lemma verticalStrip_mono {A B A' B' : ℝ} (hA : A ≤ A') (hB : B' ≤ B) :
    verticalStrip A B ⊆ verticalStrip A' B' := by
  /-
    A B A' B' : Real
    hA : LE.le A A'
    hB : LE.le B' B
    ⊢ HasSubset.Subset (UpperHalfPlane.verticalStrip A B) (UpperHalfPlane.vertical …
  -/
  rintro z ⟨hzre, hzim⟩
  /-
    case intro
    A B A' B' : Real
    hA : LE.le A A'
    hB : LE.le B' B
    z : UpperHalfPlane
    hzre : LE.le (abs z.re) A
    hzim : LE.le B z.im
    ⊢ Membership.mem (UpperHalfPlane.verticalStrip A' B') z
  -/
  exact ⟨hzre.trans hA, hB.trans hzim⟩
  /-
    🎉 no goals
  -/


@[gcongr]
lemma verticalStrip_mono_left {A A'} (h : A ≤ A') (B) : verticalStrip A B ⊆ verticalStrip A' B :=
  verticalStrip_mono h le_rfl


@[gcongr]
lemma verticalStrip_anti_right (A) {B B'} (h : B' ≤ B) : verticalStrip A B ⊆ verticalStrip A B' :=
  verticalStrip_mono le_rfl h


lemma subset_verticalStrip_of_isCompact {K : Set ℍ} (hK : IsCompact K) :
    ∃ A B : ℝ, 0 < B ∧ K ⊆ verticalStrip A B := by
  /-
    K : Set UpperHalfPlane
    hK : IsCompact K
    ⊢ Exists fun A => Exists fun B => And (LT.lt 0 B) (HasSubset.Subset K (UpperHa …
  -/
  rcases K.eq_empty_or_nonempty with rfl | hne
    /-
      case inl
      hK : IsCompact EmptyCollection.emptyCollection
      ⊢ Exists fun A => Exists fun B => And (LT.lt 0 B) (HasSubset.Subset EmptyColle …
    -/
  · exact ⟨1, 1, Real.zero_lt_one, empty_subset _⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    K : Set UpperHalfPlane
    hK : IsCompact K
    hne : K.Nonempty
    ⊢ Exists fun A => Exists fun B => And (LT.lt 0 B) (HasSubset.Subset K (UpperHa …
  -/
  obtain ⟨u, _, hu⟩ := hK.exists_isMaxOn hne (_root_.continuous_abs.comp continuous_re).continuousOn
  /-
    case inr.intro.intro
    K : Set UpperHalfPlane
    hK : IsCompact K
    hne : K.Nonempty
    u : UpperHalfPlane
    left✝ : Membership.mem K u
    hu : IsMaxOn (Function.comp abs UpperHalfPlane.re) K u
    ⊢ Exists fun A => Exists fun B => And (LT.lt 0 B) (HasSubset.Subset K (UpperHa …
  -/
  obtain ⟨v, _, hv⟩ := hK.exists_isMinOn hne continuous_im.continuousOn
  /-
    case inr.intro.intro.intro.intro
    K : Set UpperHalfPlane
    hK : IsCompact K
    hne : K.Nonempty
    u : UpperHalfPlane
    left✝¹ : Membership.mem K u
    hu : IsMaxOn (Function.comp abs UpperHalfPlane.re) K u
    v : UpperHalfPlane
    left✝ : Membership.mem K v
    hv : IsMinOn UpperHalfPlane.im K v
    ⊢ Exists fun A => Exists fun B => And (LT.lt 0 B) (HasSubset.Subset K (UpperHa …
  -/
  exact ⟨|re u|, im v, v.im_pos, fun k hk ↦ ⟨isMaxOn_iff.mp hu _ hk, isMinOn_iff.mp hv _ hk⟩⟩
  /-
    🎉 no goals
  -/


theorem ModularGroup_T_zpow_mem_verticalStrip (z : ℍ) {N : ℕ} (hn : 0 < N) :
    ∃ n : ℤ, ModularGroup.T ^ (N * n) • z ∈ verticalStrip N z.im := by
  /-
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    ⊢ Exists fun n => Membership.mem (UpperHalfPlane.verticalStrip (↑N) z.im) (HSM …
  -/
  let n := Int.floor (z.re/N)
  /-
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    ⊢ Exists fun n => Membership.mem (UpperHalfPlane.verticalStrip (↑N) z.im) (HSM …
  -/
  use -n
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    ⊢ Membership.mem (UpperHalfPlane.verticalStrip (↑N) z.im) (HSMul.hSMul (HPow.h …
  -/
  rw [modular_T_zpow_smul z (N * -n)]
  refine ⟨?_, (by simp only [mul_neg, Int.cast_neg, Int.cast_mul, Int.cast_natCast, vadd_im,
    le_refl])⟩
  have h : (N * (-n : ℝ) +ᵥ z).re = -N * Int.floor (z.re / N) + z.re := by
    simp only [n, Int.cast_natCast, mul_neg, vadd_re, neg_mul]
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (HMul.hMul (↑N) (Neg.neg ↑n)) z).re (HAdd.hAdd (HMul.hMul  …
    ⊢ LE.le (abs (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re) ↑N
  -/
  norm_cast at *
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re (HAdd.hAdd (↑(HMul.h …
    ⊢ LE.le (abs (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re) ↑N
  -/
  rw [h, add_comm]
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re (HAdd.hAdd (↑(HMul.h …
    ⊢ LE.le (abs (HAdd.hAdd z.re ↑(HMul.hMul (Neg.neg ↑N) (Int.floor (HDiv.hDiv z. …
  -/
  simp only [neg_mul, Int.cast_neg, Int.cast_mul, Int.cast_natCast]
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re (HAdd.hAdd (↑(HMul.h …
    ⊢ LE.le (abs (HAdd.hAdd z.re (Neg.neg (HMul.hMul ↑N ↑(Int.floor (HDiv.hDiv z.r …
  -/
  have hnn : (0 : ℝ) < (N : ℝ) := by norm_cast at *
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re (HAdd.hAdd (↑(HMul.h …
    hnn : LT.lt 0 ↑N
    ⊢ LE.le (abs (HAdd.hAdd z.re (Neg.neg (HMul.hMul ↑N ↑(Int.floor (HDiv.hDiv z.r …
  -/
  have h2 : z.re + -(N * n) =  z.re - n * N := by ring
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re (HAdd.hAdd (↑(HMul.h …
    hnn : LT.lt 0 ↑N
    h2 : Eq (HAdd.hAdd z.re (Neg.neg (HMul.hMul ↑N ↑n))) (HSub.hSub z.re (HMul.hMu …
    ⊢ LE.le (abs (HAdd.hAdd z.re (Neg.neg (HMul.hMul ↑N ↑(Int.floor (HDiv.hDiv z.r …
  -/
  rw [h2, abs_eq_self.2 (Int.sub_floor_div_mul_nonneg (z.re : ℝ) hnn)]
  /-
    case h
    z : UpperHalfPlane
    N : Nat
    hn : LT.lt 0 N
    n : Int := Int.floor (HDiv.hDiv z.re ↑N)
    h : Eq (HVAdd.hVAdd (↑(HMul.hMul (↑N) (Neg.neg n))) z).re (HAdd.hAdd (↑(HMul.h …
    hnn : LT.lt 0 ↑N
    h2 : Eq (HAdd.hAdd z.re (Neg.neg (HMul.hMul ↑N ↑n))) (HSub.hSub z.re (HMul.hMu …
    ⊢ LE.le (HSub.hSub z.re (HMul.hMul ↑(Int.floor (HDiv.hDiv z.re ↑N)) ↑N)) ↑N
  -/
  apply (Int.sub_floor_div_mul_lt (z.re : ℝ) hnn).le
  /-
    🎉 no goals
  -/


/-- A section `ℂ → ℍ` of the natural inclusion map, bundled as a `PartialHomeomorph`. -/
def ofComplex : PartialHomeomorph ℂ ℍ := (isOpenEmbedding_coe.toPartialHomeomorph _).symm


/-- Extend a function on `ℍ` arbitrarily to a function on all of `ℂ`. -/
scoped notation "↑ₕ" f => f ∘ ofComplex


@[simp]
lemma ofComplex_apply (z : ℍ) : ofComplex (z : ℂ) = z :=
  IsOpenEmbedding.toPartialHomeomorph_left_inv ..


lemma ofComplex_apply_eq_ite (w : ℂ) :
    ofComplex w = if hw : 0 < w.im then ⟨w, hw⟩ else Classical.choice inferInstance := by
  /-
    w : Complex
    ⊢ Eq (↑UpperHalfPlane.ofComplex w) (dite (LT.lt 0 w.im) (fun hw => ⟨w, hw⟩) fu …
  -/
  split_ifs with hw
    /-
      case pos
      w : Complex
      hw : LT.lt 0 w.im
      ⊢ Eq (↑UpperHalfPlane.ofComplex w) ⟨w, hw⟩
    -/
  · exact ofComplex_apply ⟨w, hw⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      w : Complex
      hw : Not (LT.lt 0 w.im)
      ⊢ Eq (↑UpperHalfPlane.ofComplex w) (Classical.choice ⋯)
    -/
  · change (Function.invFunOn UpperHalfPlane.coe Set.univ w) = _
    /-
      case neg
      w : Complex
      hw : Not (LT.lt 0 w.im)
      ⊢ Eq (Function.invFunOn UpperHalfPlane.coe Set.univ w) (Classical.choice ⋯)
    -/
    simp only [invFunOn, dite_eq_right_iff, mem_univ, true_and]
    /-
      case neg
      w : Complex
      hw : Not (LT.lt 0 w.im)
      ⊢ ∀ (h : Exists fun a => Eq (↑a) w), Eq (Classical.choose ⋯) (Classical.choice …
    -/
    rintro ⟨a, rfl⟩
    /-
      case neg.intro
      a : UpperHalfPlane
      hw : Not (LT.lt 0 (↑a).im)
      ⊢ Eq (Classical.choose ⋯) (Classical.choice ⋯)
    -/
    exact (a.prop.not_le (by simpa using hw)).elim
    /-
      🎉 no goals
    -/


lemma ofComplex_apply_of_im_pos {z : ℂ} (hz : 0 < z.im) :
    ofComplex z = ⟨z, hz⟩ := by
  /-
    z : Complex
    hz : LT.lt 0 z.im
    ⊢ Eq (↑UpperHalfPlane.ofComplex z) ⟨z, hz⟩
  -/
  simpa only [coe_mk_subtype] using ofComplex_apply ⟨z, hz⟩
  /-
    🎉 no goals
  -/


lemma ofComplex_apply_of_im_nonpos {w : ℂ} (hw : w.im ≤ 0) :
    ofComplex w = Classical.choice inferInstance := by
  /-
    w : Complex
    hw : LE.le w.im 0
    ⊢ Eq (↑UpperHalfPlane.ofComplex w) (Classical.choice ⋯)
  -/
  simp [ofComplex_apply_eq_ite w, hw]
  /-
    🎉 no goals
  -/


lemma ofComplex_apply_eq_of_im_nonpos {w w' : ℂ} (hw : w.im ≤ 0) (hw' : w'.im ≤ 0) :
    ofComplex w = ofComplex w' := by
  /-
    w w' : Complex
    hw : LE.le w.im 0
    hw' : LE.le w'.im 0
    ⊢ Eq (↑UpperHalfPlane.ofComplex w) (↑UpperHalfPlane.ofComplex w')
  -/
  simp [ofComplex_apply_of_im_nonpos, hw, hw']
  /-
    🎉 no goals
  -/


lemma comp_ofComplex (f : ℍ → ℂ) (z : ℍ) : (↑ₕ f) z = f z :=
  congrArg _ <| ofComplex_apply z


lemma comp_ofComplex_of_im_pos (f : ℍ → ℂ) (z : ℂ) (hz : 0 < z.im) : (↑ₕ f) z = f ⟨z, hz⟩ :=
  congrArg _ <| ofComplex_apply ⟨z, hz⟩


lemma comp_ofComplex_of_im_le_zero (f : ℍ → ℂ) (z z' : ℂ) (hz : z.im ≤ 0) (hz' : z'.im ≤ 0)  :
    (↑ₕ f) z = (↑ₕ f) z' := by
  /-
    f : UpperHalfPlane → Complex
    z z' : Complex
    hz : LE.le z.im 0
    hz' : LE.le z'.im 0
    ⊢ Eq (Function.comp f (↑UpperHalfPlane.ofComplex) z) (Function.comp f (↑UpperH …
  -/
  simp [ofComplex_apply_of_im_nonpos, hz, hz']
  /-
    🎉 no goals
  -/


lemma eventuallyEq_coe_comp_ofComplex {z : ℂ} (hz : 0 < z.im) :
    UpperHalfPlane.coe ∘ ofComplex =ᶠ[𝓝 z] id := by
  /-
    z : Complex
    hz : LT.lt 0 z.im
    ⊢ (nhds z).EventuallyEq (Function.comp UpperHalfPlane.coe ↑UpperHalfPlane.ofCo …
  -/
  filter_upwards [(Complex.continuous_im.isOpen_preimage _ isOpen_Ioi).mem_nhds hz] with x hx
  /-
    case h
    z : Complex
    hz : LT.lt 0 z.im
    x : Complex
    hx : Membership.mem (Set.preimage Complex.im (Set.Ioi 0)) x
    ⊢ Eq (Function.comp UpperHalfPlane.coe (↑UpperHalfPlane.ofComplex) x) (id x)
  -/
  simp only [Function.comp_apply, ofComplex_apply_of_im_pos hx, id_eq, coe_mk_subtype]
  /-
    🎉 no goals
  -/


