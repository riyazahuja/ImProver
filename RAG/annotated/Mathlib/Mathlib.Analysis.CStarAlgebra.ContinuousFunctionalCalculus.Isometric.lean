local notation "σ" => spectrum

local notation "σₙ" => quasispectrum


/-- An extension of the `ContinuousFunctionalCalculus` requiring that `cfcHom` is an isometry. -/
class IsometricContinuousFunctionalCalculus (R A : Type*) (p : outParam (A → Prop))
    [CommSemiring R] [StarRing R] [MetricSpace R] [TopologicalSemiring R] [ContinuousStar R]
    [Ring A] [StarRing A] [MetricSpace A] [Algebra R A]
    extends ContinuousFunctionalCalculus R p : Prop where
  isometric (a : A) (ha : p a) : Isometry (cfcHom ha (R := R))


lemma isometry_cfcHom {R A : Type*} {p : outParam (A → Prop)} [CommSemiring R] [StarRing R]
    [MetricSpace R] [TopologicalSemiring R] [ContinuousStar R] [Ring A] [StarRing A]
    [MetricSpace A] [Algebra R A] [IsometricContinuousFunctionalCalculus R A p]
    (a : A) (ha : p a := by cfc_tac) :
    Isometry (cfcHom (show p a from ha) (R := R)) :=
  IsometricContinuousFunctionalCalculus.isometric a ha


lemma norm_cfcHom (a : A) (f : C(σ 𝕜 a, 𝕜)) (ha : p a := by cfc_tac) :
    ‖cfcHom (show p a from ha) f‖ = ‖f‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    a : A
    f : ContinuousMap (↑(spectrum 𝕜 a)) 𝕜
    ha : autoParam (p a) _auto✝
    ⊢ Eq (Norm.norm ((cfcHom ⋯) f)) (Norm.norm f)
  -/
  refine isometry_cfcHom a |>.norm_map_of_map_zero (map_zero _) f
  /-
    🎉 no goals
  -/


lemma nnnorm_cfcHom (a : A) (f : C(σ 𝕜 a, 𝕜)) (ha : p a := by cfc_tac) :
    ‖cfcHom (show p a from ha) f‖₊ = ‖f‖₊ :=
  Subtype.ext <| norm_cfcHom a f ha


lemma IsGreatest.norm_cfc [Nontrivial A] (f : 𝕜 → 𝕜) (a : A)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    IsGreatest ((fun x ↦ ‖f x‖) '' spectrum 𝕜 a) ‖cfc f a‖ := by
  obtain ⟨x, hx⟩ := ContinuousFunctionalCalculus.isCompact_spectrum a
    |>.image_of_continuousOn hf.norm |>.exists_isGreatest <|
    (ContinuousFunctionalCalculus.spectrum_nonempty a ha).image _
  /-
    case intro
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
    inst✝ : Nontrivial A
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    x : Real
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) x
    ⊢ IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) (Norm.norm  …
  -/
  obtain ⟨x, hx', rfl⟩ := hx.1
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
    inst✝ : Nontrivial A
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    x : 𝕜
    hx' : Membership.mem (spectrum 𝕜 a) x
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
    ⊢ IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) (Norm.norm  …
  -/
  convert hx
  /-
    case h.e'_4
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
    inst✝ : Nontrivial A
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    x : 𝕜
    hx' : Membership.mem (spectrum 𝕜 a) x
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
    ⊢ Eq (Norm.norm (cfc f a)) ((fun x => Norm.norm (f x)) x)
  -/
  rw [cfc_apply f a, norm_cfcHom a _]
  /-
    case h.e'_4
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
    inst✝ : Nontrivial A
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    x : 𝕜
    hx' : Membership.mem (spectrum 𝕜 a) x
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
    ⊢ Eq (Norm.norm { toFun := (spectrum 𝕜 a).restrict f, continuous_toFun := ⋯ }) …
  -/
  apply le_antisymm
    /-
      case h.e'_4.a
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedRing A
      inst✝³ : StarRing A
      inst✝² : NormedAlgebra 𝕜 A
      inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
      inst✝ : Nontrivial A
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (spectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
      ⊢ LE.le (Norm.norm { toFun := (spectrum 𝕜 a).restrict f, continuous_toFun := ⋯ …
    -/
  · apply ContinuousMap.norm_le _ (norm_nonneg _) |>.mpr
    /-
      case h.e'_4.a
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedRing A
      inst✝³ : StarRing A
      inst✝² : NormedAlgebra 𝕜 A
      inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
      inst✝ : Nontrivial A
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (spectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
      ⊢ ∀ (x_1 : ↑(spectrum 𝕜 a)), LE.le (Norm.norm ({ toFun := (spectrum 𝕜 a).restr …
    -/
    rintro ⟨y, hy⟩
    /-
      case h.e'_4.a.mk
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedRing A
      inst✝³ : StarRing A
      inst✝² : NormedAlgebra 𝕜 A
      inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
      inst✝ : Nontrivial A
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (spectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
      y : 𝕜
      hy : Membership.mem (spectrum 𝕜 a) y
      ⊢ LE.le (Norm.norm ({ toFun := (spectrum 𝕜 a).restrict f, continuous_toFun :=  …
    -/
    exact hx.2 ⟨y, hy, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.a
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedRing A
      inst✝³ : StarRing A
      inst✝² : NormedAlgebra 𝕜 A
      inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
      inst✝ : Nontrivial A
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (spectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (spectrum 𝕜 a)) ((fun x  …
      ⊢ LE.le ((fun x => Norm.norm (f x)) x) (Norm.norm { toFun := (spectrum 𝕜 a).re …
    -/
  · exact le_trans (by simp) <| ContinuousMap.norm_coe_le_norm _ (⟨x, hx'⟩ : σ 𝕜 a)
    /-
      🎉 no goals
    -/


lemma IsGreatest.nnnorm_cfc [Nontrivial A] (f : 𝕜 → 𝕜) (a : A)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    IsGreatest ((fun x ↦ ‖f x‖₊) '' σ 𝕜 a) ‖cfc f a‖₊ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
    inst✝ : Nontrivial A
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ IsGreatest (Set.image (fun x => NNNorm.nnnorm (f x)) (spectrum 𝕜 a)) (NNNorm …
  -/
  convert Real.toNNReal_mono.map_isGreatest (.norm_cfc f a)
  /-
    case h.e'_3
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedRing A
    inst✝³ : StarRing A
    inst✝² : NormedAlgebra 𝕜 A
    inst✝¹ : IsometricContinuousFunctionalCalculus 𝕜 A p
    inst✝ : Nontrivial A
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (Set.image (fun x => NNNorm.nnnorm (f x)) (spectrum 𝕜 a)) (Set.image Real …
  -/
  all_goals simp [Set.image_image, norm_toNNReal]
  /-
    🎉 no goals
  -/


lemma norm_apply_le_norm_cfc (f : 𝕜 → 𝕜) (a : A) ⦃x : 𝕜⦄ (hx : x ∈ σ 𝕜 a)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    ‖f x‖ ≤ ‖cfc f a‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    x : 𝕜
    hx : Membership.mem (spectrum 𝕜 a) x
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ LE.le (Norm.norm (f x)) (Norm.norm (cfc f a))
  -/
  revert hx
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    x : 𝕜
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) (Norm.norm (cfc f  …
  -/
  nontriviality A
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    x : 𝕜
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    a✝ : Nontrivial A
    ⊢ Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) (Norm.norm (cfc f  …
  -/
  exact (IsGreatest.norm_cfc f a hf ha |>.2 ⟨x, ·, rfl⟩)
  /-
    🎉 no goals
  -/


lemma nnnorm_apply_le_nnnorm_cfc (f : 𝕜 → 𝕜) (a : A) ⦃x : 𝕜⦄ (hx : x ∈ σ 𝕜 a)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac) (ha : p a := by cfc_tac) :
    ‖f x‖₊ ≤ ‖cfc f a‖₊ :=
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    x : 𝕜
    hx : Membership.mem (spectrum 𝕜 a) x
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ ContinuousOn f (spectrum 𝕜 a)
  -/
  /-
    🎉 no goals
  -/
  norm_apply_le_norm_cfc f a hx
  /-
    🎉 no goals
  -/


lemma norm_cfc_le {f : 𝕜 → 𝕜} {a : A} {c : ℝ} (hc : 0 ≤ c) (h : ∀ x ∈ σ 𝕜 a, ‖f x‖ ≤ c) :
    ‖cfc f a‖ ≤ c := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    c : Real
    hc : LE.le 0 c
    h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
    ⊢ LE.le (Norm.norm (cfc f a)) c
  -/
  obtain (_ | _) := subsingleton_or_nontrivial A
    /-
      case inl
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LE.le 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      h✝ : Subsingleton A
      ⊢ LE.le (Norm.norm (cfc f a)) c
    -/
  · simpa [Subsingleton.elim (cfc f a) 0]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LE.le 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      h✝ : Nontrivial A
      ⊢ LE.le (Norm.norm (cfc f a)) c
    -/
  · refine cfc_cases (‖·‖ ≤ c) a f (by simpa) fun hf ha ↦ ?_
    /-
      case inr
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LE.le 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum 𝕜 a)
      ha : p a
      ⊢ (fun x => LE.le (Norm.norm x) c) ((cfcHom ha) { toFun := (spectrum 𝕜 a).rest …
    -/
    simp only [← cfc_apply f a, isLUB_le_iff (IsGreatest.norm_cfc f a hf ha |>.isLUB)]
    /-
      case inr
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LE.le 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum 𝕜 a)
      ha : p a
      ⊢ Membership.mem (upperBounds (Set.image (fun x => Norm.norm (f x)) (spectrum  …
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case inr.intro.intro
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LE.le 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum 𝕜 a)
      ha : p a
      x : 𝕜
      hx : Membership.mem (spectrum 𝕜 a) x
      ⊢ LE.le ((fun x => Norm.norm (f x)) x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma norm_cfc_le_iff (f : 𝕜 → 𝕜) (a : A) {c : ℝ} (hc : 0 ≤ c)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : ‖cfc f a‖ ≤ c ↔ ∀ x ∈ σ 𝕜 a, ‖f x‖ ≤ c :=
  ⟨fun h _ hx ↦ norm_apply_le_norm_cfc f a hx hf ha |>.trans h, norm_cfc_le hc⟩


lemma norm_cfc_lt {f : 𝕜 → 𝕜} {a : A} {c : ℝ} (hc : 0 < c) (h : ∀ x ∈ σ 𝕜 a, ‖f x‖ < c) :
    ‖cfc f a‖ < c := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    c : Real
    hc : LT.lt 0 c
    h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
    ⊢ LT.lt (Norm.norm (cfc f a)) c
  -/
  obtain (_ | _) := subsingleton_or_nontrivial A
    /-
      case inl
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LT.lt 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      h✝ : Subsingleton A
      ⊢ LT.lt (Norm.norm (cfc f a)) c
    -/
  · simpa [Subsingleton.elim (cfc f a) 0]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LT.lt 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      h✝ : Nontrivial A
      ⊢ LT.lt (Norm.norm (cfc f a)) c
    -/
  · refine cfc_cases (‖·‖ < c) a f (by simpa) fun hf ha ↦ ?_
    /-
      case inr
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LT.lt 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum 𝕜 a)
      ha : p a
      ⊢ (fun x => LT.lt (Norm.norm x) c) ((cfcHom ha) { toFun := (spectrum 𝕜 a).rest …
    -/
    simp only [← cfc_apply f a, (IsGreatest.norm_cfc f a hf ha |>.lt_iff)]
    /-
      case inr
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LT.lt 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum 𝕜 a)
      ha : p a
      ⊢ ∀ (x : Real), Membership.mem (Set.image (fun x => Norm.norm (f x)) (spectrum …
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case inr.intro.intro
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedRing A
      inst✝² : StarRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      hc : LT.lt 0 c
      h : ∀ (x : 𝕜), Membership.mem (spectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum 𝕜 a)
      ha : p a
      x : 𝕜
      hx : Membership.mem (spectrum 𝕜 a) x
      ⊢ LT.lt ((fun x => Norm.norm (f x)) x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma norm_cfc_lt_iff (f : 𝕜 → 𝕜) (a : A) {c : ℝ} (hc : 0 < c)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : ‖cfc f a‖ < c ↔ ∀ x ∈ σ 𝕜 a, ‖f x‖ < c :=
  ⟨fun h _ hx ↦ norm_apply_le_norm_cfc f a hx hf ha |>.trans_lt h, norm_cfc_lt hc⟩


lemma nnnorm_cfc_le {f : 𝕜 → 𝕜} {a : A} (c : ℝ≥0) (h : ∀ x ∈ σ 𝕜 a, ‖f x‖₊ ≤ c) :
    ‖cfc f a‖₊ ≤ c :=
  norm_cfc_le c.2 h


lemma nnnorm_cfc_le_iff (f : 𝕜 → 𝕜) (a : A) (c : ℝ≥0)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : ‖cfc f a‖₊ ≤ c ↔ ∀ x ∈ σ 𝕜 a, ‖f x‖₊ ≤ c :=
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    c : NNReal
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ ContinuousOn f (spectrum 𝕜 a)
  -/
  /-
    🎉 no goals
  -/
  norm_cfc_le_iff f a c.2
  /-
    🎉 no goals
  -/


lemma nnnorm_cfc_lt {f : 𝕜 → 𝕜} {a : A} {c : ℝ≥0} (hc : 0 < c) (h : ∀ x ∈ σ 𝕜 a, ‖f x‖₊ < c) :
    ‖cfc f a‖₊ < c :=
  norm_cfc_lt hc h


lemma nnnorm_cfc_lt_iff (f : 𝕜 → 𝕜) (a : A) {c : ℝ≥0} (hc : 0 < c)
    (hf : ContinuousOn f (σ 𝕜 a) := by cfc_cont_tac)
    (ha : p a := by cfc_tac) : ‖cfc f a‖₊ < c ↔ ∀ x ∈ σ 𝕜 a, ‖f x‖₊ < c :=
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedRing A
    inst✝² : StarRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    c : NNReal
    hc : LT.lt 0 c
    hf : autoParam (ContinuousOn f (spectrum 𝕜 a)) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ ContinuousOn f (spectrum 𝕜 a)
  -/
  /-
    🎉 no goals
  -/
  norm_cfc_lt_iff f a hc
  /-
    🎉 no goals
  -/


open scoped ContinuousFunctionalCalculus in
protected theorem isometric_cfc (f : C(S, R)) (halg : Isometry (algebraMap R S)) (h0 : p 0)
    (h : ∀ a, p a ↔ q a ∧ SpectrumRestricts a f) :
    IsometricContinuousFunctionalCalculus R A p where
  toContinuousFunctionalCalculus := SpectrumRestricts.cfc f halg.isUniformEmbedding h0 h
  isometric a ha := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ⊢ Isometry ⇑(cfcHom ha)
    -/
    obtain ⟨ha', haf⟩ := h a |>.mp ha
    have _inst (a : A) : CompactSpace (σ R a) := by
      rw [← isCompact_iff_compactSpace, ← spectrum.preimage_algebraMap S]
      exact halg.isClosedEmbedding.isCompact_preimage <|
        ContinuousFunctionalCalculus.isCompact_spectrum a
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : SpectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
      ⊢ Isometry ⇑(cfcHom ha)
    -/
    have := SpectrumRestricts.cfc f halg.isUniformEmbedding h0 h
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : SpectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
      this : ContinuousFunctionalCalculus R p
      ⊢ Isometry ⇑(cfcHom ha)
    -/
    rw [cfcHom_eq_restrict f halg.isUniformEmbedding ha ha' haf]
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : SpectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
      this : ContinuousFunctionalCalculus R p
      ⊢ Isometry ⇑(SpectrumRestricts.starAlgHom (cfcHom ha') haf)
    -/
    refine .of_dist_eq fun g₁ g₂ ↦ ?_
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : SpectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
      this : ContinuousFunctionalCalculus R p
      g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
      ⊢ Eq (Dist.dist ((SpectrumRestricts.starAlgHom (cfcHom ha') haf) g₁) ((Spectru …
    -/
    simp only [starAlgHom_apply, isometry_cfcHom a ha' |>.dist_eq]
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : SpectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
      this : ContinuousFunctionalCalculus R p
      g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
      ⊢ Eq (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯ }.co …
    -/
    refine le_antisymm ?_ ?_
    /-
      case intro.refine_1
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²¹ : Semifield R
      inst✝²⁰ : StarRing R
      inst✝¹⁹ : MetricSpace R
      inst✝¹⁸ : TopologicalSemiring R
      inst✝¹⁷ : ContinuousStar R
      inst✝¹⁶ : Semifield S
      inst✝¹⁵ : StarRing S
      inst✝¹⁴ : MetricSpace S
      inst✝¹³ : TopologicalSemiring S
      inst✝¹² : ContinuousStar S
      inst✝¹¹ : Ring A
      inst✝¹⁰ : StarRing A
      inst✝⁹ : Algebra S A
      inst✝⁸ : Algebra R S
      inst✝⁷ : Algebra R A
      inst✝⁶ : IsScalarTower R S A
      inst✝⁵ : StarModule R S
      inst✝⁴ : ContinuousSMul R S
      inst✝³ : MetricSpace A
      inst✝² : IsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : SpectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
      this : ContinuousFunctionalCalculus R p
      g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
      ⊢ LE.le (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯ } …
    -/
    all_goals refine ContinuousMap.dist_le dist_nonneg |>.mpr fun x ↦ ?_
      /-
        case intro.refine_1
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²¹ : Semifield R
        inst✝²⁰ : StarRing R
        inst✝¹⁹ : MetricSpace R
        inst✝¹⁸ : TopologicalSemiring R
        inst✝¹⁷ : ContinuousStar R
        inst✝¹⁶ : Semifield S
        inst✝¹⁵ : StarRing S
        inst✝¹⁴ : MetricSpace S
        inst✝¹³ : TopologicalSemiring S
        inst✝¹² : ContinuousStar S
        inst✝¹¹ : Ring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra S A
        inst✝⁸ : Algebra R S
        inst✝⁷ : Algebra R A
        inst✝⁶ : IsScalarTower R S A
        inst✝⁵ : StarModule R S
        inst✝⁴ : ContinuousSMul R S
        inst✝³ : MetricSpace A
        inst✝² : IsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : SpectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
        this : ContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
        x : ↑(spectrum S a)
        ⊢ LE.le (Dist.dist (({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯  …
      -/
    · simpa [halg.dist_eq] using ContinuousMap.dist_apply_le_dist _
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²¹ : Semifield R
        inst✝²⁰ : StarRing R
        inst✝¹⁹ : MetricSpace R
        inst✝¹⁸ : TopologicalSemiring R
        inst✝¹⁷ : ContinuousStar R
        inst✝¹⁶ : Semifield S
        inst✝¹⁵ : StarRing S
        inst✝¹⁴ : MetricSpace S
        inst✝¹³ : TopologicalSemiring S
        inst✝¹² : ContinuousStar S
        inst✝¹¹ : Ring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra S A
        inst✝⁸ : Algebra R S
        inst✝⁷ : Algebra R A
        inst✝⁶ : IsScalarTower R S A
        inst✝⁵ : StarModule R S
        inst✝⁴ : ContinuousSMul R S
        inst✝³ : MetricSpace A
        inst✝² : IsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : SpectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
        this : ContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
        x : ↑(spectrum R a)
        ⊢ LE.le (Dist.dist (g₁ x) (g₂ x)) (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R S …
      -/
    · let x' : σ S a := Subtype.map (algebraMap R S) (fun _ ↦ spectrum.algebraMap_mem S) x
      /-
        case intro.refine_2
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²¹ : Semifield R
        inst✝²⁰ : StarRing R
        inst✝¹⁹ : MetricSpace R
        inst✝¹⁸ : TopologicalSemiring R
        inst✝¹⁷ : ContinuousStar R
        inst✝¹⁶ : Semifield S
        inst✝¹⁵ : StarRing S
        inst✝¹⁴ : MetricSpace S
        inst✝¹³ : TopologicalSemiring S
        inst✝¹² : ContinuousStar S
        inst✝¹¹ : Ring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra S A
        inst✝⁸ : Algebra R S
        inst✝⁷ : Algebra R A
        inst✝⁶ : IsScalarTower R S A
        inst✝⁵ : StarModule R S
        inst✝⁴ : ContinuousSMul R S
        inst✝³ : MetricSpace A
        inst✝² : IsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : SpectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
        this : ContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
        x : ↑(spectrum R a)
        x' : ↑(spectrum S a) := Subtype.map ⇑(algebraMap R S) ⋯ x
        ⊢ LE.le (Dist.dist (g₁ x) (g₂ x)) (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R S …
      -/
      apply le_of_eq_of_le ?_ <| ContinuousMap.dist_apply_le_dist x'
      simp only [ContinuousMap.comp_apply, ContinuousMap.coe_mk, StarAlgHom.ofId_apply,
        halg.dist_eq, x']
      /-
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²¹ : Semifield R
        inst✝²⁰ : StarRing R
        inst✝¹⁹ : MetricSpace R
        inst✝¹⁸ : TopologicalSemiring R
        inst✝¹⁷ : ContinuousStar R
        inst✝¹⁶ : Semifield S
        inst✝¹⁵ : StarRing S
        inst✝¹⁴ : MetricSpace S
        inst✝¹³ : TopologicalSemiring S
        inst✝¹² : ContinuousStar S
        inst✝¹¹ : Ring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra S A
        inst✝⁸ : Algebra R S
        inst✝⁷ : Algebra R A
        inst✝⁶ : IsScalarTower R S A
        inst✝⁵ : StarModule R S
        inst✝⁴ : ContinuousSMul R S
        inst✝³ : MetricSpace A
        inst✝² : IsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : SpectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
        this : ContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
        x : ↑(spectrum R a)
        x' : ↑(spectrum S a) := Subtype.map ⇑(algebraMap R S) ⋯ x
        ⊢ Eq (Dist.dist (g₁ x) (g₂ x)) (Dist.dist (g₁ (Subtype.map ⇑f ⋯ (Subtype.map ⇑ …
      -/
      congr!
      /-
        case h.e'_3.h.e'_6
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²¹ : Semifield R
        inst✝²⁰ : StarRing R
        inst✝¹⁹ : MetricSpace R
        inst✝¹⁸ : TopologicalSemiring R
        inst✝¹⁷ : ContinuousStar R
        inst✝¹⁶ : Semifield S
        inst✝¹⁵ : StarRing S
        inst✝¹⁴ : MetricSpace S
        inst✝¹³ : TopologicalSemiring S
        inst✝¹² : ContinuousStar S
        inst✝¹¹ : Ring A
        inst✝¹⁰ : StarRing A
        inst✝⁹ : Algebra S A
        inst✝⁸ : Algebra R S
        inst✝⁷ : Algebra R A
        inst✝⁶ : IsScalarTower R S A
        inst✝⁵ : StarModule R S
        inst✝⁴ : ContinuousSMul R S
        inst✝³ : MetricSpace A
        inst✝² : IsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (SpectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : SpectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(spectrum R a)
        this : ContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMap (↑(spectrum R a)) R
        x : ↑(spectrum R a)
        x' : ↑(spectrum S a) := Subtype.map ⇑(algebraMap R S) ⋯ x
        ⊢ Eq x (Subtype.map ⇑f ⋯ (Subtype.map ⇑(algebraMap R S) ⋯ x))
      -/
      all_goals ext; exact haf.left_inv _ |>.symm
      /-
        🎉 no goals
      -/


/-- An extension of the `NonUnitalContinuousFunctionalCalculus` requiring that `cfcₙHom` is an
isometry. -/
class NonUnitalIsometricContinuousFunctionalCalculus (R A : Type*) (p : outParam (A → Prop))
    [CommSemiring R] [Nontrivial R] [StarRing R] [MetricSpace R] [TopologicalSemiring R]
    [ContinuousStar R] [NonUnitalRing A] [StarRing A] [MetricSpace A] [Module R A]
    [IsScalarTower R A A] [SMulCommClass R A A]
    extends NonUnitalContinuousFunctionalCalculus R p : Prop where
  isometric (a : A) (ha : p a) : Isometry (cfcₙHom ha (R := R))


lemma isometry_cfcₙHom (a : A) (ha : p a := by cfc_tac) :
    Isometry (cfcₙHom (show p a from ha) (R := R)) :=
  NonUnitalIsometricContinuousFunctionalCalculus.isometric a ha


lemma norm_cfcₙHom (a : A) (f : C(σₙ 𝕜 a, 𝕜)₀) (ha : p a := by cfc_tac) :
    ‖cfcₙHom (show p a from ha) f‖ = ‖f‖ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    a : A
    f : ContinuousMapZero (↑(quasispectrum 𝕜 a)) 𝕜
    ha : autoParam (p a) _auto✝
    ⊢ Eq (Norm.norm ((cfcₙHom ⋯) f)) (Norm.norm f)
  -/
  refine isometry_cfcₙHom a |>.norm_map_of_map_zero (map_zero _) f
  /-
    🎉 no goals
  -/


lemma nnnorm_cfcₙHom (a : A) (f : C(σₙ 𝕜 a, 𝕜)₀) (ha : p a := by cfc_tac) :
    ‖cfcₙHom (show p a from ha) f‖₊ = ‖f‖₊ :=
  Subtype.ext <| norm_cfcₙHom a f ha


lemma IsGreatest.norm_cfcₙ (f : 𝕜 → 𝕜) (a : A)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : IsGreatest ((fun x ↦ ‖f x‖) '' σₙ 𝕜 a) ‖cfcₙ f a‖ := by
  obtain ⟨x, hx⟩ := NonUnitalContinuousFunctionalCalculus.isCompact_quasispectrum a
      |>.image_of_continuousOn hf.norm |>.exists_isGreatest <|
      (quasispectrum.nonempty 𝕜 a).image _
  /-
    case intro
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (p a) _auto✝
    x : Real
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) x
    ⊢ IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) (Norm. …
  -/
  obtain ⟨x, hx', rfl⟩ := hx.1
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (p a) _auto✝
    x : 𝕜
    hx' : Membership.mem (quasispectrum 𝕜 a) x
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
    ⊢ IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) (Norm. …
  -/
  convert hx
  /-
    case h.e'_4
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (p a) _auto✝
    x : 𝕜
    hx' : Membership.mem (quasispectrum 𝕜 a) x
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
    ⊢ Eq (Norm.norm (cfcₙ f a)) ((fun x => Norm.norm (f x)) x)
  -/
  rw [cfcₙ_apply f a, norm_cfcₙHom a _]
  /-
    case h.e'_4
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (p a) _auto✝
    x : 𝕜
    hx' : Membership.mem (quasispectrum 𝕜 a) x
    hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
    ⊢ Eq (Norm.norm { toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := …
  -/
  apply le_antisymm
    /-
      case h.e'_4.a
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (quasispectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
      ⊢ LE.le (Norm.norm { toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun …
    -/
  · apply ContinuousMap.norm_le _ (norm_nonneg _) |>.mpr
    /-
      case h.e'_4.a
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (quasispectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
      ⊢ ∀ (x_1 : ↑(quasispectrum 𝕜 a)), LE.le (Norm.norm (↑{ toFun := (quasispectrum …
    -/
    rintro ⟨y, hy⟩
    /-
      case h.e'_4.a.mk
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (quasispectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
      y : 𝕜
      hy : Membership.mem (quasispectrum 𝕜 a) y
      ⊢ LE.le (Norm.norm (↑{ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toF …
    -/
    exact hx.2 ⟨y, hy, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.a
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
      hf₀ : autoParam (Eq (f 0) 0) _auto✝
      ha : autoParam (p a) _auto✝
      x : 𝕜
      hx' : Membership.mem (quasispectrum 𝕜 a) x
      hx : IsGreatest (Set.image (fun x => Norm.norm (f x)) (quasispectrum 𝕜 a)) ((f …
      ⊢ LE.le ((fun x => Norm.norm (f x)) x) (Norm.norm { toFun := (quasispectrum 𝕜  …
    -/
  · exact le_trans (by simp) <| ContinuousMap.norm_coe_le_norm _ (⟨x, hx'⟩ : σₙ 𝕜 a)
    /-
      🎉 no goals
    -/


lemma IsGreatest.nnnorm_cfcₙ (f : 𝕜 → 𝕜) (a : A)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : IsGreatest ((fun x ↦ ‖f x‖₊) '' σₙ 𝕜 a) ‖cfcₙ f a‖₊ := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ IsGreatest (Set.image (fun x => NNNorm.nnnorm (f x)) (quasispectrum 𝕜 a)) (N …
  -/
  convert Real.toNNReal_mono.map_isGreatest (.norm_cfcₙ f a)
  /-
    case h.e'_3
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum 𝕜 a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (p a) _auto✝
    ⊢ Eq (Set.image (fun x => NNNorm.nnnorm (f x)) (quasispectrum 𝕜 a)) (Set.image …
  -/
  all_goals simp [Set.image_image, norm_toNNReal]
  /-
    🎉 no goals
  -/


lemma norm_apply_le_norm_cfcₙ (f : 𝕜 → 𝕜) (a : A) ⦃x : 𝕜⦄ (hx : x ∈ σₙ 𝕜 a)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : ‖f x‖ ≤ ‖cfcₙ f a‖ :=
  IsGreatest.norm_cfcₙ f a hf hf₀ ha |>.2 ⟨x, hx, rfl⟩


lemma nnnorm_apply_le_nnnorm_cfcₙ (f : 𝕜 → 𝕜) (a : A) ⦃x : 𝕜⦄ (hx : x ∈ σₙ 𝕜 a)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : ‖f x‖₊ ≤ ‖cfcₙ f a‖₊ :=
  IsGreatest.nnnorm_cfcₙ f a hf hf₀ ha |>.2 ⟨x, hx, rfl⟩


lemma norm_cfcₙ_le {f : 𝕜 → 𝕜} {a : A} {c : ℝ} (h : ∀ x ∈ σₙ 𝕜 a, ‖f x‖ ≤ c) :
    ‖cfcₙ f a‖ ≤ c := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    c : Real
    h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
    ⊢ LE.le (Norm.norm (cfcₙ f a)) c
  -/
  refine cfcₙ_cases (‖·‖ ≤ c) a f ?_ fun hf hf0 ha ↦ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      ⊢ (fun x => LE.le (Norm.norm x) c) 0
    -/
  · simpa using (norm_nonneg _).trans <| h 0 (quasispectrum.zero_mem 𝕜 a)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      hf : ContinuousOn f (quasispectrum 𝕜 a)
      hf0 : Eq ({ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := ⋯ } 0 …
      ha : p a
      ⊢ (fun x => LE.le (Norm.norm x) c) ((cfcₙHom ha) { toFun := (quasispectrum 𝕜 a …
    -/
  · simp only [← cfcₙ_apply f a, isLUB_le_iff (IsGreatest.norm_cfcₙ f a hf hf0 ha |>.isLUB)]
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      hf : ContinuousOn f (quasispectrum 𝕜 a)
      hf0 : Eq ({ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := ⋯ } 0 …
      ha : p a
      ⊢ Membership.mem (upperBounds (Set.image (fun x => Norm.norm (f x)) (quasispec …
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case refine_2.intro.intro
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LE.le (Norm.norm (f x)) c
      hf : ContinuousOn f (quasispectrum 𝕜 a)
      hf0 : Eq ({ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := ⋯ } 0 …
      ha : p a
      x : 𝕜
      hx : Membership.mem (quasispectrum 𝕜 a) x
      ⊢ LE.le ((fun x => Norm.norm (f x)) x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma norm_cfcₙ_le_iff (f : 𝕜 → 𝕜) (a : A) (c : ℝ)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : ‖cfcₙ f a‖ ≤ c ↔ ∀ x ∈ σₙ 𝕜 a, ‖f x‖ ≤ c :=
  ⟨fun h _ hx ↦ norm_apply_le_norm_cfcₙ f a hx hf hf₀ ha |>.trans h, norm_cfcₙ_le⟩


lemma norm_cfcₙ_lt {f : 𝕜 → 𝕜} {a : A} {c : ℝ} (h : ∀ x ∈ σₙ 𝕜 a, ‖f x‖ < c) :
    ‖cfcₙ f a‖ < c := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    p : outParam (A → Prop)
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NonUnitalNormedRing A
    inst✝⁴ : StarRing A
    inst✝³ : NormedSpace 𝕜 A
    inst✝² : IsScalarTower 𝕜 A A
    inst✝¹ : SMulCommClass 𝕜 A A
    inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
    f : 𝕜 → 𝕜
    a : A
    c : Real
    h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
    ⊢ LT.lt (Norm.norm (cfcₙ f a)) c
  -/
  refine cfcₙ_cases (‖·‖ < c) a f ?_ fun hf hf0 ha ↦ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      ⊢ (fun x => LT.lt (Norm.norm x) c) 0
    -/
  · simpa using (norm_nonneg _).trans_lt <| h 0 (quasispectrum.zero_mem 𝕜 a)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      hf : ContinuousOn f (quasispectrum 𝕜 a)
      hf0 : Eq ({ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := ⋯ } 0 …
      ha : p a
      ⊢ (fun x => LT.lt (Norm.norm x) c) ((cfcₙHom ha) { toFun := (quasispectrum 𝕜 a …
    -/
  · simp only [← cfcₙ_apply f a, (IsGreatest.norm_cfcₙ f a hf hf0 ha |>.lt_iff)]
    /-
      case refine_2
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      hf : ContinuousOn f (quasispectrum 𝕜 a)
      hf0 : Eq ({ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := ⋯ } 0 …
      ha : p a
      ⊢ ∀ (x : Real), Membership.mem (Set.image (fun x => Norm.norm (f x)) (quasispe …
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case refine_2.intro.intro
      𝕜 : Type u_1
      A : Type u_2
      p : outParam (A → Prop)
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NonUnitalNormedRing A
      inst✝⁴ : StarRing A
      inst✝³ : NormedSpace 𝕜 A
      inst✝² : IsScalarTower 𝕜 A A
      inst✝¹ : SMulCommClass 𝕜 A A
      inst✝ : NonUnitalIsometricContinuousFunctionalCalculus 𝕜 A p
      f : 𝕜 → 𝕜
      a : A
      c : Real
      h : ∀ (x : 𝕜), Membership.mem (quasispectrum 𝕜 a) x → LT.lt (Norm.norm (f x)) c
      hf : ContinuousOn f (quasispectrum 𝕜 a)
      hf0 : Eq ({ toFun := (quasispectrum 𝕜 a).restrict f, continuous_toFun := ⋯ } 0 …
      ha : p a
      x : 𝕜
      hx : Membership.mem (quasispectrum 𝕜 a) x
      ⊢ LT.lt ((fun x => Norm.norm (f x)) x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma norm_cfcₙ_lt_iff (f : 𝕜 → 𝕜) (a : A) (c : ℝ)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : ‖cfcₙ f a‖ < c ↔ ∀ x ∈ σₙ 𝕜 a, ‖f x‖ < c :=
  ⟨fun h _ hx ↦ norm_apply_le_norm_cfcₙ f a hx hf hf₀ ha |>.trans_lt h, norm_cfcₙ_lt⟩


lemma nnnorm_cfcₙ_le {f : 𝕜 → 𝕜} {a : A} {c : ℝ≥0} (h : ∀ x ∈ σₙ 𝕜 a, ‖f x‖₊ ≤ c) :
    ‖cfcₙ f a‖₊ ≤ c :=
  norm_cfcₙ_le h


lemma nnnorm_cfcₙ_le_iff (f : 𝕜 → 𝕜) (a : A) (c : ℝ≥0)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : ‖cfcₙ f a‖₊ ≤ c ↔ ∀ x ∈ σₙ 𝕜 a, ‖f x‖₊ ≤ c :=
  norm_cfcₙ_le_iff f a c.1 hf hf₀ ha


lemma nnnorm_cfcₙ_lt {f : 𝕜 → 𝕜} {a : A} {c : ℝ≥0} (h : ∀ x ∈ σₙ 𝕜 a, ‖f x‖₊ < c) :
    ‖cfcₙ f a‖₊ < c :=
  norm_cfcₙ_lt h


lemma nnnorm_cfcₙ_lt_iff (f : 𝕜 → 𝕜) (a : A) (c : ℝ≥0)
    (hf : ContinuousOn f (σₙ 𝕜 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : p a := by cfc_tac) : ‖cfcₙ f a‖₊ < c ↔ ∀ x ∈ σₙ 𝕜 a, ‖f x‖₊ < c :=
  norm_cfcₙ_lt_iff f a c.1 hf hf₀ ha


open scoped NonUnitalContinuousFunctionalCalculus in
protected theorem isometric_cfc (f : C(S, R)) (halg : Isometry (algebraMap R S)) (h0 : p 0)
    (h : ∀ a, p a ↔ q a ∧ QuasispectrumRestricts a f) :
    NonUnitalIsometricContinuousFunctionalCalculus R A p where
  toNonUnitalContinuousFunctionalCalculus := QuasispectrumRestricts.cfc f
    halg.isUniformEmbedding h0 h
  isometric a ha := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ⊢ Isometry ⇑(cfcₙHom ha)
    -/
    obtain ⟨ha', haf⟩ := h a |>.mp ha
    have _inst (a : A) : CompactSpace (σₙ R a) := by
      rw [← isCompact_iff_compactSpace, ← quasispectrum.preimage_algebraMap S]
      exact halg.isClosedEmbedding.isCompact_preimage <|
        NonUnitalContinuousFunctionalCalculus.isCompact_quasispectrum a
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : QuasispectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
      ⊢ Isometry ⇑(cfcₙHom ha)
    -/
    have := QuasispectrumRestricts.cfc f halg.isUniformEmbedding h0 h
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : QuasispectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
      this : NonUnitalContinuousFunctionalCalculus R p
      ⊢ Isometry ⇑(cfcₙHom ha)
    -/
    rw [cfcₙHom_eq_restrict f halg.isUniformEmbedding ha ha' haf]
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : QuasispectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
      this : NonUnitalContinuousFunctionalCalculus R p
      ⊢ Isometry ⇑(QuasispectrumRestricts.nonUnitalStarAlgHom (cfcₙHom ha') haf)
    -/
    refine .of_dist_eq fun g₁ g₂ ↦ ?_
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : QuasispectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
      this : NonUnitalContinuousFunctionalCalculus R p
      g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
      ⊢ Eq (Dist.dist ((QuasispectrumRestricts.nonUnitalStarAlgHom (cfcₙHom ha') haf …
    -/
    simp only [nonUnitalStarAlgHom_apply, isometry_cfcₙHom a ha' |>.dist_eq]
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : QuasispectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
      this : NonUnitalContinuousFunctionalCalculus R p
      g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
      ⊢ Eq (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯, map …
    -/
    refine le_antisymm ?_ ?_
    /-
      case intro.refine_1
      R : Type u_1
      S : Type u_2
      A : Type u_3
      p q : A → Prop
      inst✝²⁵ : Semifield R
      inst✝²⁴ : StarRing R
      inst✝²³ : MetricSpace R
      inst✝²² : TopologicalSemiring R
      inst✝²¹ : ContinuousStar R
      inst✝²⁰ : Field S
      inst✝¹⁹ : StarRing S
      inst✝¹⁸ : MetricSpace S
      inst✝¹⁷ : TopologicalRing S
      inst✝¹⁶ : ContinuousStar S
      inst✝¹⁵ : NonUnitalRing A
      inst✝¹⁴ : StarRing A
      inst✝¹³ : Module S A
      inst✝¹² : IsScalarTower S A A
      inst✝¹¹ : SMulCommClass S A A
      inst✝¹⁰ : Algebra R S
      inst✝⁹ : Module R A
      inst✝⁸ : IsScalarTower R S A
      inst✝⁷ : StarModule R S
      inst✝⁶ : ContinuousSMul R S
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : SMulCommClass R A A
      inst✝³ : MetricSpace A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
      inst✝¹ : CompleteSpace R
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
      f : ContinuousMap S R
      halg : Isometry ⇑(algebraMap R S)
      h0 : p 0
      h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
      a : A
      ha : p a
      ha' : q a
      haf : QuasispectrumRestricts a ⇑f
      _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
      this : NonUnitalContinuousFunctionalCalculus R p
      g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
      ⊢ LE.le (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯,  …
    -/
    all_goals refine ContinuousMap.dist_le dist_nonneg |>.mpr fun x ↦ ?_
      /-
        case intro.refine_1
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁵ : Semifield R
        inst✝²⁴ : StarRing R
        inst✝²³ : MetricSpace R
        inst✝²² : TopologicalSemiring R
        inst✝²¹ : ContinuousStar R
        inst✝²⁰ : Field S
        inst✝¹⁹ : StarRing S
        inst✝¹⁸ : MetricSpace S
        inst✝¹⁷ : TopologicalRing S
        inst✝¹⁶ : ContinuousStar S
        inst✝¹⁵ : NonUnitalRing A
        inst✝¹⁴ : StarRing A
        inst✝¹³ : Module S A
        inst✝¹² : IsScalarTower S A A
        inst✝¹¹ : SMulCommClass S A A
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R A
        inst✝⁸ : IsScalarTower R S A
        inst✝⁷ : StarModule R S
        inst✝⁶ : ContinuousSMul R S
        inst✝⁵ : IsScalarTower R A A
        inst✝⁴ : SMulCommClass R A A
        inst✝³ : MetricSpace A
        inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : QuasispectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
        this : NonUnitalContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
        x : ↑(quasispectrum S a)
        ⊢ LE.le (Dist.dist (↑({ toFun := ⇑(StarAlgHom.ofId R S), continuous_toFun := ⋯ …
      -/
    · simpa [halg.dist_eq] using ContinuousMap.dist_apply_le_dist _
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁵ : Semifield R
        inst✝²⁴ : StarRing R
        inst✝²³ : MetricSpace R
        inst✝²² : TopologicalSemiring R
        inst✝²¹ : ContinuousStar R
        inst✝²⁰ : Field S
        inst✝¹⁹ : StarRing S
        inst✝¹⁸ : MetricSpace S
        inst✝¹⁷ : TopologicalRing S
        inst✝¹⁶ : ContinuousStar S
        inst✝¹⁵ : NonUnitalRing A
        inst✝¹⁴ : StarRing A
        inst✝¹³ : Module S A
        inst✝¹² : IsScalarTower S A A
        inst✝¹¹ : SMulCommClass S A A
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R A
        inst✝⁸ : IsScalarTower R S A
        inst✝⁷ : StarModule R S
        inst✝⁶ : ContinuousSMul R S
        inst✝⁵ : IsScalarTower R A A
        inst✝⁴ : SMulCommClass R A A
        inst✝³ : MetricSpace A
        inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : QuasispectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
        this : NonUnitalContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
        x : ↑(quasispectrum R a)
        ⊢ LE.le (Dist.dist (↑g₁ x) (↑g₂ x)) (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R …
      -/
    · let x' : σₙ S a := Subtype.map (algebraMap R S) (fun _ ↦ quasispectrum.algebraMap_mem S) x
      /-
        case intro.refine_2
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁵ : Semifield R
        inst✝²⁴ : StarRing R
        inst✝²³ : MetricSpace R
        inst✝²² : TopologicalSemiring R
        inst✝²¹ : ContinuousStar R
        inst✝²⁰ : Field S
        inst✝¹⁹ : StarRing S
        inst✝¹⁸ : MetricSpace S
        inst✝¹⁷ : TopologicalRing S
        inst✝¹⁶ : ContinuousStar S
        inst✝¹⁵ : NonUnitalRing A
        inst✝¹⁴ : StarRing A
        inst✝¹³ : Module S A
        inst✝¹² : IsScalarTower S A A
        inst✝¹¹ : SMulCommClass S A A
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R A
        inst✝⁸ : IsScalarTower R S A
        inst✝⁷ : StarModule R S
        inst✝⁶ : ContinuousSMul R S
        inst✝⁵ : IsScalarTower R A A
        inst✝⁴ : SMulCommClass R A A
        inst✝³ : MetricSpace A
        inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : QuasispectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
        this : NonUnitalContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
        x : ↑(quasispectrum R a)
        x' : ↑(quasispectrum S a) := Subtype.map ⇑(algebraMap R S) ⋯ x
        ⊢ LE.le (Dist.dist (↑g₁ x) (↑g₂ x)) (Dist.dist ({ toFun := ⇑(StarAlgHom.ofId R …
      -/
      apply le_of_eq_of_le ?_ <| ContinuousMap.dist_apply_le_dist x'
      simp only [ContinuousMap.coe_coe, ContinuousMapZero.comp_apply, ContinuousMapZero.coe_mk,
        ContinuousMap.coe_mk, StarAlgHom.ofId_apply, halg.dist_eq, x']
      /-
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁵ : Semifield R
        inst✝²⁴ : StarRing R
        inst✝²³ : MetricSpace R
        inst✝²² : TopologicalSemiring R
        inst✝²¹ : ContinuousStar R
        inst✝²⁰ : Field S
        inst✝¹⁹ : StarRing S
        inst✝¹⁸ : MetricSpace S
        inst✝¹⁷ : TopologicalRing S
        inst✝¹⁶ : ContinuousStar S
        inst✝¹⁵ : NonUnitalRing A
        inst✝¹⁴ : StarRing A
        inst✝¹³ : Module S A
        inst✝¹² : IsScalarTower S A A
        inst✝¹¹ : SMulCommClass S A A
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R A
        inst✝⁸ : IsScalarTower R S A
        inst✝⁷ : StarModule R S
        inst✝⁶ : ContinuousSMul R S
        inst✝⁵ : IsScalarTower R A A
        inst✝⁴ : SMulCommClass R A A
        inst✝³ : MetricSpace A
        inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : QuasispectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
        this : NonUnitalContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
        x : ↑(quasispectrum R a)
        x' : ↑(quasispectrum S a) := Subtype.map ⇑(algebraMap R S) ⋯ x
        ⊢ Eq (Dist.dist (g₁ x) (g₂ x)) (Dist.dist (g₁ (Subtype.map ⇑f ⋯ (Subtype.map ⇑ …
      -/
      congr! 2
      /-
        case h.e'_3.h.e'_6
        R : Type u_1
        S : Type u_2
        A : Type u_3
        p q : A → Prop
        inst✝²⁵ : Semifield R
        inst✝²⁴ : StarRing R
        inst✝²³ : MetricSpace R
        inst✝²² : TopologicalSemiring R
        inst✝²¹ : ContinuousStar R
        inst✝²⁰ : Field S
        inst✝¹⁹ : StarRing S
        inst✝¹⁸ : MetricSpace S
        inst✝¹⁷ : TopologicalRing S
        inst✝¹⁶ : ContinuousStar S
        inst✝¹⁵ : NonUnitalRing A
        inst✝¹⁴ : StarRing A
        inst✝¹³ : Module S A
        inst✝¹² : IsScalarTower S A A
        inst✝¹¹ : SMulCommClass S A A
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R A
        inst✝⁸ : IsScalarTower R S A
        inst✝⁷ : StarModule R S
        inst✝⁶ : ContinuousSMul R S
        inst✝⁵ : IsScalarTower R A A
        inst✝⁴ : SMulCommClass R A A
        inst✝³ : MetricSpace A
        inst✝² : NonUnitalIsometricContinuousFunctionalCalculus S A q
        inst✝¹ : CompleteSpace R
        inst✝ : UniqueNonUnitalContinuousFunctionalCalculus R A
        f : ContinuousMap S R
        halg : Isometry ⇑(algebraMap R S)
        h0 : p 0
        h : ∀ (a : A), Iff (p a) (And (q a) (QuasispectrumRestricts a ⇑f))
        a : A
        ha : p a
        ha' : q a
        haf : QuasispectrumRestricts a ⇑f
        _inst : ∀ (a : A), CompactSpace ↑(quasispectrum R a)
        this : NonUnitalContinuousFunctionalCalculus R p
        g₁ g₂ : ContinuousMapZero (↑(quasispectrum R a)) R
        x : ↑(quasispectrum R a)
        x' : ↑(quasispectrum S a) := Subtype.map ⇑(algebraMap R S) ⋯ x
        ⊢ Eq x (Subtype.map ⇑f ⋯ (Subtype.map ⇑(algebraMap R S) ⋯ x))
      -/
      all_goals ext; exact haf.left_inv _ |>.symm
      /-
        🎉 no goals
      -/


instance IsStarNormal.instIsometricContinuousFunctionalCalculus :
    IsometricContinuousFunctionalCalculus ℂ A IsStarNormal where
  isometric a ha := by
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      a : A
      ha : IsStarNormal a
      ⊢ Isometry ⇑(cfcHom ha)
    -/
    rw [cfcHom_eq_of_isStarNormal]
    /-
      A : Type u_1
      inst✝ : CStarAlgebra A
      a : A
      ha : IsStarNormal a
      ⊢ Isometry ⇑((StarAlgebra.elemental Complex a).subtype.comp ↑(continuousFuncti …
    -/
    exact isometry_subtype_coe.comp <| StarAlgEquiv.isometry (continuousFunctionalCalculus a)
    /-
      🎉 no goals
    -/


instance IsSelfAdjoint.instIsometricContinuousFunctionalCalculus :
    IsometricContinuousFunctionalCalculus ℝ A IsSelfAdjoint :=
  SpectrumRestricts.isometric_cfc Complex.reCLM Complex.isometry_ofReal (.zero _)
    fun _ ↦ isSelfAdjoint_iff_isStarNormal_and_spectrumRestricts


open NNReal in
instance Nonneg.instIsometricContinuousFunctionalCalculus :
    IsometricContinuousFunctionalCalculus ℝ≥0 A (0 ≤ ·) :=
  SpectrumRestricts.isometric_cfc (q := IsSelfAdjoint) ContinuousMap.realToNNReal
    isometry_subtype_coe le_rfl (fun _ ↦ nonneg_iff_isSelfAdjoint_and_spectrumRestricts)


open ContinuousMapZero in
instance IsStarNormal.instNonUnitalIsometricContinuousFunctionalCalculus :
    NonUnitalIsometricContinuousFunctionalCalculus ℂ A IsStarNormal where
  isometric a ha := by
    /-
      A : Type u_1
      inst✝ : NonUnitalCStarAlgebra A
      a : A
      ha : IsStarNormal a
      ⊢ Isometry ⇑(cfcₙHom ha)
    -/
    refine AddMonoidHomClass.isometry_of_norm _ fun f ↦ ?_
    rw [← norm_inr (𝕜 := ℂ), ← inrNonUnitalStarAlgHom_apply, ← NonUnitalStarAlgHom.comp_apply,
      inr_comp_cfcₙHom_eq_cfcₙAux a, cfcₙAux]
    simp only [NonUnitalStarAlgHom.comp_assoc, NonUnitalStarAlgHom.comp_apply,
      toContinuousMapHom_apply, NonUnitalStarAlgHom.coe_coe]
    /-
      A : Type u_1
      inst✝ : NonUnitalCStarAlgebra A
      a : A
      ha : IsStarNormal a
      f : ContinuousMapZero (↑(quasispectrum Complex a)) Complex
      ⊢ Eq (Norm.norm ((cfcHom ⋯) ((Homeomorph.compStarAlgEquiv' Complex Complex (Ho …
    -/
    rw [norm_cfcHom (a : Unitization ℂ A), StarAlgEquiv.norm_map]
    /-
      A : Type u_1
      inst✝ : NonUnitalCStarAlgebra A
      a : A
      ha : IsStarNormal a
      f : ContinuousMapZero (↑(quasispectrum Complex a)) Complex
      ⊢ Eq (Norm.norm ↑f) (Norm.norm f)
    -/
    rfl
    /-
      🎉 no goals
    -/


instance IsSelfAdjoint.instNonUnitalIsometricContinuousFunctionalCalculus :
    NonUnitalIsometricContinuousFunctionalCalculus ℝ A IsSelfAdjoint :=
  QuasispectrumRestricts.isometric_cfc Complex.reCLM Complex.isometry_ofReal (.zero _)
    fun _ ↦ isSelfAdjoint_iff_isStarNormal_and_quasispectrumRestricts


open NNReal in
instance Nonneg.instNonUnitalIsometricContinuousFunctionalCalculus :
    NonUnitalIsometricContinuousFunctionalCalculus ℝ≥0 A (0 ≤ ·) :=
  QuasispectrumRestricts.isometric_cfc (q := IsSelfAdjoint) ContinuousMap.realToNNReal
    isometry_subtype_coe le_rfl (fun _ ↦ nonneg_iff_isSelfAdjoint_and_quasispectrumRestricts)


lemma IsGreatest.nnnorm_cfc_nnreal [Nontrivial A] (f : ℝ≥0 → ℝ≥0) (a : A)
    (hf : ContinuousOn f (σ ℝ≥0 a) := by cfc_cont_tac) (ha : 0 ≤ a := by cfc_tac) :
    IsGreatest (f '' σ ℝ≥0 a) ‖cfc f a‖₊ := by
  /-
    A : Type u_1
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝² : NonnegSpectrumClass Real A
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : Nontrivial A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ IsGreatest (Set.image f (spectrum NNReal a)) (NNNorm.nnnorm (cfc f a))
  -/
  rw [cfc_nnreal_eq_real]
  /-
    A : Type u_1
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝² : NonnegSpectrumClass Real A
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : Nontrivial A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ IsGreatest (Set.image f (spectrum NNReal a)) (NNNorm.nnnorm (cfc (fun x => ↑ …
  -/
  obtain ⟨-, ha'⟩ := nonneg_iff_isSelfAdjoint_and_spectrumRestricts.mp ha
  /-
    case intro
    A : Type u_1
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝² : NonnegSpectrumClass Real A
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : Nontrivial A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ IsGreatest (Set.image f (spectrum NNReal a)) (NNNorm.nnnorm (cfc (fun x => ↑ …
  -/
  convert IsGreatest.nnnorm_cfc (fun x : ℝ ↦ (f x.toNNReal : ℝ)) a ?hf_cont
  case hf_cont => exact continuous_subtype_val.comp_continuousOn <|
    ContinuousOn.comp ‹_› continuous_real_toNNReal.continuousOn <| ha'.image ▸ Set.mapsTo_image ..
  /-
    case h.e'_3
    A : Type u_1
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝² : NonnegSpectrumClass Real A
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : Nontrivial A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ Eq (Set.image f (spectrum NNReal a)) (Set.image (fun x => NNNorm.nnnorm ↑(f  …
  -/
  ext x
  /-
    case h.e'_3.h
    A : Type u_1
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝² : NonnegSpectrumClass Real A
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : Nontrivial A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : NNReal
    ⊢ Iff (Membership.mem (Set.image f (spectrum NNReal a)) x) (Membership.mem (Se …
  -/
  constructor
  /-
    case h.e'_3.h.mp
    A : Type u_1
    inst✝⁸ : NormedRing A
    inst✝⁷ : StarRing A
    inst✝⁶ : NormedAlgebra Real A
    inst✝⁵ : PartialOrder A
    inst✝⁴ : StarOrderedRing A
    inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝² : NonnegSpectrumClass Real A
    inst✝¹ : UniqueContinuousFunctionalCalculus Real A
    inst✝ : Nontrivial A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : NNReal
    ⊢ Membership.mem (Set.image f (spectrum NNReal a)) x → Membership.mem (Set.ima …
  -/
  all_goals rintro ⟨x, hx, rfl⟩
    /-
      case h.e'_3.h.mp.intro.intro
      A : Type u_1
      inst✝⁸ : NormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedAlgebra Real A
      inst✝⁵ : PartialOrder A
      inst✝⁴ : StarOrderedRing A
      inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝² : NonnegSpectrumClass Real A
      inst✝¹ : UniqueContinuousFunctionalCalculus Real A
      inst✝ : Nontrivial A
      f : NNReal → NNReal
      a : A
      hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
      ha : autoParam (LE.le 0 a) _auto✝
      ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : NNReal
      hx : Membership.mem (spectrum NNReal a) x
      ⊢ Membership.mem (Set.image (fun x => NNNorm.nnnorm ↑(f x.toNNReal)) (spectrum …
    -/
  · exact ⟨x, spectrum.algebraMap_mem ℝ hx, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.mpr.intro.intro
      A : Type u_1
      inst✝⁸ : NormedRing A
      inst✝⁷ : StarRing A
      inst✝⁶ : NormedAlgebra Real A
      inst✝⁵ : PartialOrder A
      inst✝⁴ : StarOrderedRing A
      inst✝³ : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝² : NonnegSpectrumClass Real A
      inst✝¹ : UniqueContinuousFunctionalCalculus Real A
      inst✝ : Nontrivial A
      f : NNReal → NNReal
      a : A
      hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
      ha : autoParam (LE.le 0 a) _auto✝
      ha' : SpectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : Real
      hx : Membership.mem (spectrum Real a) x
      ⊢ Membership.mem (Set.image f (spectrum NNReal a)) ((fun x => NNNorm.nnnorm ↑( …
    -/
  · exact ⟨x.toNNReal, ha'.apply_mem hx, by simp⟩
    /-
      🎉 no goals
    -/


lemma apply_le_nnnorm_cfc_nnreal (f : ℝ≥0 → ℝ≥0) (a : A) ⦃x : ℝ≥0⦄ (hx : x ∈ σ ℝ≥0 a)
    (hf : ContinuousOn f (σ ℝ≥0 a) := by cfc_cont_tac) (ha : 0 ≤ a := by cfc_tac) :
    f x ≤ ‖cfc f a‖₊ := by
  /-
    A : Type u_1
    inst✝⁷ : NormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    x : NNReal
    hx : Membership.mem (spectrum NNReal a) x
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ LE.le (f x) (NNNorm.nnnorm (cfc f a))
  -/
  revert hx
  /-
    A : Type u_1
    inst✝⁷ : NormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    x : NNReal
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Membership.mem (spectrum NNReal a) x → LE.le (f x) (NNNorm.nnnorm (cfc f a))
  -/
  nontriviality A
  /-
    A : Type u_1
    inst✝⁷ : NormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    x : NNReal
    hf : autoParam (ContinuousOn f (spectrum NNReal a)) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    a✝ : Nontrivial A
    ⊢ Membership.mem (spectrum NNReal a) x → LE.le (f x) (NNNorm.nnnorm (cfc f a))
  -/
  exact (IsGreatest.nnnorm_cfc_nnreal f a hf ha |>.2 ⟨x, ·, rfl⟩)
  /-
    🎉 no goals
  -/


lemma nnnorm_cfc_nnreal_le {f : ℝ≥0 → ℝ≥0} {a : A} {c : ℝ≥0} (h : ∀ x ∈ σ ℝ≥0 a, f x ≤ c) :
    ‖cfc f a‖₊ ≤ c := by
  /-
    A : Type u_1
    inst✝⁷ : NormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
    ⊢ LE.le (NNNorm.nnnorm (cfc f a)) c
  -/
  obtain (_ | _) := subsingleton_or_nontrivial A
    /-
      case inl
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
      h✝ : Subsingleton A
      ⊢ LE.le (NNNorm.nnnorm (cfc f a)) c
    -/
  · rw [Subsingleton.elim (cfc f a) 0]
    /-
      case inl
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
      h✝ : Subsingleton A
      ⊢ LE.le (NNNorm.nnnorm 0) c
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
      h✝ : Nontrivial A
      ⊢ LE.le (NNNorm.nnnorm (cfc f a)) c
    -/
  · refine cfc_cases (‖·‖₊ ≤ c) a f (by simp) fun hf ha ↦ ?_
    /-
      case inr
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum NNReal a)
      ha : LE.le 0 a
      ⊢ (fun x => LE.le (NNNorm.nnnorm x) c) ((cfcHom ha) { toFun := (spectrum NNRea …
    -/
    simp only [← cfc_apply f a, isLUB_le_iff (IsGreatest.nnnorm_cfc_nnreal f a hf ha |>.isLUB)]
    /-
      case inr
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum NNReal a)
      ha : LE.le 0 a
      ⊢ Membership.mem (upperBounds (Set.image f (spectrum NNReal a))) c
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case inr.intro.intro
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LE.le (f x) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum NNReal a)
      ha : LE.le 0 a
      x : NNReal
      hx : Membership.mem (spectrum NNReal a) x
      ⊢ LE.le (f x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma nnnorm_cfc_nnreal_le_iff (f : ℝ≥0 → ℝ≥0) (a : A) (c : ℝ≥0)
    (hf : ContinuousOn f (σ ℝ≥0 a) := by cfc_cont_tac)
    (ha : 0 ≤ a := by cfc_tac) : ‖cfc f a‖₊ ≤ c ↔ ∀ x ∈ σ ℝ≥0 a, f x ≤ c :=
  ⟨fun h _ hx ↦ apply_le_nnnorm_cfc_nnreal f a hx hf ha |>.trans h, nnnorm_cfc_nnreal_le⟩


lemma nnnorm_cfc_nnreal_lt {f : ℝ≥0 → ℝ≥0} {a : A} {c : ℝ≥0} (hc : 0 < c)
    (h : ∀ x ∈ σ ℝ≥0 a, f x < c) : ‖cfc f a‖₊ < c := by
  /-
    A : Type u_1
    inst✝⁷ : NormedRing A
    inst✝⁶ : StarRing A
    inst✝⁵ : NormedAlgebra Real A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
    ⊢ LT.lt (NNNorm.nnnorm (cfc f a)) c
  -/
  obtain (_ | _) := subsingleton_or_nontrivial A
    /-
      case inl
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      hc : LT.lt 0 c
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
      h✝ : Subsingleton A
      ⊢ LT.lt (NNNorm.nnnorm (cfc f a)) c
    -/
  · rw [Subsingleton.elim (cfc f a) 0]
    /-
      case inl
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      hc : LT.lt 0 c
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
      h✝ : Subsingleton A
      ⊢ LT.lt (NNNorm.nnnorm 0) c
    -/
    simpa
    /-
      🎉 no goals
    -/
    /-
      case inr
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      hc : LT.lt 0 c
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
      h✝ : Nontrivial A
      ⊢ LT.lt (NNNorm.nnnorm (cfc f a)) c
    -/
  · refine cfc_cases (‖·‖₊ < c) a f (by simpa) fun hf ha ↦ ?_
    /-
      case inr
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      hc : LT.lt 0 c
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum NNReal a)
      ha : LE.le 0 a
      ⊢ (fun x => LT.lt (NNNorm.nnnorm x) c) ((cfcHom ha) { toFun := (spectrum NNRea …
    -/
    simp only [← cfc_apply f a, (IsGreatest.nnnorm_cfc_nnreal f a hf ha |>.lt_iff)]
    /-
      case inr
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      hc : LT.lt 0 c
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum NNReal a)
      ha : LE.le 0 a
      ⊢ ∀ (x : NNReal), Membership.mem (Set.image f (spectrum NNReal a)) x → LT.lt x c
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case inr.intro.intro
      A : Type u_1
      inst✝⁷ : NormedRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : NormedAlgebra Real A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : IsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      hc : LT.lt 0 c
      h : ∀ (x : NNReal), Membership.mem (spectrum NNReal a) x → LT.lt (f x) c
      h✝ : Nontrivial A
      hf : ContinuousOn f (spectrum NNReal a)
      ha : LE.le 0 a
      x : NNReal
      hx : Membership.mem (spectrum NNReal a) x
      ⊢ LT.lt (f x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma nnnorm_cfc_nnreal_lt_iff (f : ℝ≥0 → ℝ≥0) (a : A) {c : ℝ≥0} (hc : 0 < c)
    (hf : ContinuousOn f (σ ℝ≥0 a) := by cfc_cont_tac)
    (ha : 0 ≤ a := by cfc_tac) : ‖cfc f a‖₊ < c ↔ ∀ x ∈ σ ℝ≥0 a, f x < c :=
  ⟨fun h _ hx ↦ apply_le_nnnorm_cfc_nnreal f a hx hf ha |>.trans_lt h, nnnorm_cfc_nnreal_lt hc⟩


lemma IsGreatest.nnnorm_cfcₙ_nnreal (f : ℝ≥0 → ℝ≥0) (a : A)
    (hf : ContinuousOn f (σₙ ℝ≥0 a) := by cfc_cont_tac) (hf0 : f 0 = 0 := by cfc_zero_tac)
    (ha : 0 ≤ a := by cfc_tac) : IsGreatest (f '' σₙ ℝ≥0 a) ‖cfcₙ f a‖₊ := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ IsGreatest (Set.image f (quasispectrum NNReal a)) (NNNorm.nnnorm (cfcₙ f a))
  -/
  rw [cfcₙ_nnreal_eq_real]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ IsGreatest (Set.image f (quasispectrum NNReal a)) (NNNorm.nnnorm (cfcₙ (fun  …
  -/
  obtain ⟨-, ha'⟩ := nonneg_iff_isSelfAdjoint_and_quasispectrumRestricts.mp ha
  /-
    case intro
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ IsGreatest (Set.image f (quasispectrum NNReal a)) (NNNorm.nnnorm (cfcₙ (fun  …
  -/
  convert IsGreatest.nnnorm_cfcₙ (fun x : ℝ ↦ (f x.toNNReal : ℝ)) a ?hf_cont (by simpa)
  case hf_cont => exact continuous_subtype_val.comp_continuousOn <|
    ContinuousOn.comp ‹_› continuous_real_toNNReal.continuousOn <| ha'.image ▸ Set.mapsTo_image ..
  /-
    case h.e'_3
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    ⊢ Eq (Set.image f (quasispectrum NNReal a)) (Set.image (fun x => NNNorm.nnnorm …
  -/
  ext x
  /-
    case h.e'_3.h
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : NNReal
    ⊢ Iff (Membership.mem (Set.image f (quasispectrum NNReal a)) x) (Membership.me …
  -/
  constructor
  /-
    case h.e'_3.h.mp
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ha' : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
    x : NNReal
    ⊢ Membership.mem (Set.image f (quasispectrum NNReal a)) x → Membership.mem (Se …
  -/
  all_goals rintro ⟨x, hx, rfl⟩
    /-
      case h.e'_3.h.mp.intro.intro
      A : Type u_1
      inst✝⁹ : NonUnitalNormedRing A
      inst✝⁸ : StarRing A
      inst✝⁷ : NormedSpace Real A
      inst✝⁶ : IsScalarTower Real A A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
      hf0 : autoParam (Eq (f 0) 0) _auto✝
      ha : autoParam (LE.le 0 a) _auto✝
      ha' : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : NNReal
      hx : Membership.mem (quasispectrum NNReal a) x
      ⊢ Membership.mem (Set.image (fun x => NNNorm.nnnorm ↑(f x.toNNReal)) (quasispe …
    -/
  · exact ⟨x, quasispectrum.algebraMap_mem ℝ hx, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.mpr.intro.intro
      A : Type u_1
      inst✝⁹ : NonUnitalNormedRing A
      inst✝⁸ : StarRing A
      inst✝⁷ : NormedSpace Real A
      inst✝⁶ : IsScalarTower Real A A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
      hf0 : autoParam (Eq (f 0) 0) _auto✝
      ha : autoParam (LE.le 0 a) _auto✝
      ha' : QuasispectrumRestricts a ⇑ContinuousMap.realToNNReal
      x : Real
      hx : Membership.mem (quasispectrum Real a) x
      ⊢ Membership.mem (Set.image f (quasispectrum NNReal a)) ((fun x => NNNorm.nnno …
    -/
  · exact ⟨x.toNNReal, ha'.apply_mem hx, by simp⟩
    /-
      🎉 no goals
    -/


lemma apply_le_nnnorm_cfcₙ_nnreal (f : ℝ≥0 → ℝ≥0) (a : A) ⦃x : ℝ≥0⦄ (hx : x ∈ σₙ ℝ≥0 a)
    (hf : ContinuousOn f (σₙ ℝ≥0 a) := by cfc_cont_tac) (hf0 : f 0 = 0 := by cfc_zero_tac)
    (ha : 0 ≤ a := by cfc_tac) : f x ≤ ‖cfcₙ f a‖₊ := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    x : NNReal
    hx : Membership.mem (quasispectrum NNReal a) x
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ LE.le (f x) (NNNorm.nnnorm (cfcₙ f a))
  -/
  revert hx
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    x : NNReal
    hf : autoParam (ContinuousOn f (quasispectrum NNReal a)) _auto✝
    hf0 : autoParam (Eq (f 0) 0) _auto✝
    ha : autoParam (LE.le 0 a) _auto✝
    ⊢ Membership.mem (quasispectrum NNReal a) x → LE.le (f x) (NNNorm.nnnorm (cfcₙ …
  -/
  exact (IsGreatest.nnnorm_cfcₙ_nnreal f a hf hf0 ha |>.2 ⟨x, ·, rfl⟩)
  /-
    🎉 no goals
  -/


lemma nnnorm_cfcₙ_nnreal_le {f : ℝ≥0 → ℝ≥0} {a : A} {c : ℝ≥0} (h : ∀ x ∈ σₙ ℝ≥0 a, f x ≤ c) :
    ‖cfcₙ f a‖₊ ≤ c := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LE.le (f x) c
    ⊢ LE.le (NNNorm.nnnorm (cfcₙ f a)) c
  -/
  refine cfcₙ_cases (‖·‖₊ ≤ c) a f (by simp) fun hf hf0 ha ↦ ?_
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LE.le (f x) c
    hf : ContinuousOn f (quasispectrum NNReal a)
    hf0 : Eq ({ toFun := (quasispectrum NNReal a).restrict f, continuous_toFun :=  …
    ha : LE.le 0 a
    ⊢ (fun x => LE.le (NNNorm.nnnorm x) c) ((cfcₙHom ha) { toFun := (quasispectrum …
  -/
  simp only [← cfcₙ_apply f a, isLUB_le_iff (IsGreatest.nnnorm_cfcₙ_nnreal f a hf hf0 ha |>.isLUB)]
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LE.le (f x) c
    hf : ContinuousOn f (quasispectrum NNReal a)
    hf0 : Eq ({ toFun := (quasispectrum NNReal a).restrict f, continuous_toFun :=  …
    ha : LE.le 0 a
    ⊢ Membership.mem (upperBounds (Set.image f (quasispectrum NNReal a))) c
  -/
  rintro - ⟨x, hx, rfl⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LE.le (f x) c
    hf : ContinuousOn f (quasispectrum NNReal a)
    hf0 : Eq ({ toFun := (quasispectrum NNReal a).restrict f, continuous_toFun :=  …
    ha : LE.le 0 a
    x : NNReal
    hx : Membership.mem (quasispectrum NNReal a) x
    ⊢ LE.le (f x) c
  -/
  exact h x hx
  /-
    🎉 no goals
  -/


lemma nnnorm_cfcₙ_nnreal_le_iff (f : ℝ≥0 → ℝ≥0) (a : A) (c : ℝ≥0)
    (hf : ContinuousOn f (σₙ ℝ≥0 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : 0 ≤ a := by cfc_tac) : ‖cfcₙ f a‖₊ ≤ c ↔ ∀ x ∈ σₙ ℝ≥0 a, f x ≤ c :=
  ⟨fun h _ hx ↦ apply_le_nnnorm_cfcₙ_nnreal f a hx hf hf₀ ha |>.trans h, nnnorm_cfcₙ_nnreal_le⟩


lemma nnnorm_cfcₙ_nnreal_lt {f : ℝ≥0 → ℝ≥0} {a : A} {c : ℝ≥0} (h : ∀ x ∈ σₙ ℝ≥0 a, f x < c) :
    ‖cfcₙ f a‖₊ < c := by
  /-
    A : Type u_1
    inst✝⁹ : NonUnitalNormedRing A
    inst✝⁸ : StarRing A
    inst✝⁷ : NormedSpace Real A
    inst✝⁶ : IsScalarTower Real A A
    inst✝⁵ : SMulCommClass Real A A
    inst✝⁴ : PartialOrder A
    inst✝³ : StarOrderedRing A
    inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
    inst✝¹ : NonnegSpectrumClass Real A
    inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
    f : NNReal → NNReal
    a : A
    c : NNReal
    h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LT.lt (f x) c
    ⊢ LT.lt (NNNorm.nnnorm (cfcₙ f a)) c
  -/
  refine cfcₙ_cases (‖·‖₊ < c) a f ?_ fun hf hf0 ha ↦ ?_
    /-
      case refine_1
      A : Type u_1
      inst✝⁹ : NonUnitalNormedRing A
      inst✝⁸ : StarRing A
      inst✝⁷ : NormedSpace Real A
      inst✝⁶ : IsScalarTower Real A A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LT.lt (f x) c
      ⊢ (fun x => LT.lt (NNNorm.nnnorm x) c) 0
    -/
  · simpa using zero_le (f 0) |>.trans_lt <| h 0 (quasispectrum.zero_mem ℝ≥0 _)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝⁹ : NonUnitalNormedRing A
      inst✝⁸ : StarRing A
      inst✝⁷ : NormedSpace Real A
      inst✝⁶ : IsScalarTower Real A A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LT.lt (f x) c
      hf : ContinuousOn f (quasispectrum NNReal a)
      hf0 : Eq ({ toFun := (quasispectrum NNReal a).restrict f, continuous_toFun :=  …
      ha : LE.le 0 a
      ⊢ (fun x => LT.lt (NNNorm.nnnorm x) c) ((cfcₙHom ha) { toFun := (quasispectrum …
    -/
  · simp only [← cfcₙ_apply f a, (IsGreatest.nnnorm_cfcₙ_nnreal f a hf hf0 ha |>.lt_iff)]
    /-
      case refine_2
      A : Type u_1
      inst✝⁹ : NonUnitalNormedRing A
      inst✝⁸ : StarRing A
      inst✝⁷ : NormedSpace Real A
      inst✝⁶ : IsScalarTower Real A A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LT.lt (f x) c
      hf : ContinuousOn f (quasispectrum NNReal a)
      hf0 : Eq ({ toFun := (quasispectrum NNReal a).restrict f, continuous_toFun :=  …
      ha : LE.le 0 a
      ⊢ ∀ (x : NNReal), Membership.mem (Set.image f (quasispectrum NNReal a)) x → LT …
    -/
    rintro - ⟨x, hx, rfl⟩
    /-
      case refine_2.intro.intro
      A : Type u_1
      inst✝⁹ : NonUnitalNormedRing A
      inst✝⁸ : StarRing A
      inst✝⁷ : NormedSpace Real A
      inst✝⁶ : IsScalarTower Real A A
      inst✝⁵ : SMulCommClass Real A A
      inst✝⁴ : PartialOrder A
      inst✝³ : StarOrderedRing A
      inst✝² : NonUnitalIsometricContinuousFunctionalCalculus Real A IsSelfAdjoint
      inst✝¹ : NonnegSpectrumClass Real A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      f : NNReal → NNReal
      a : A
      c : NNReal
      h : ∀ (x : NNReal), Membership.mem (quasispectrum NNReal a) x → LT.lt (f x) c
      hf : ContinuousOn f (quasispectrum NNReal a)
      hf0 : Eq ({ toFun := (quasispectrum NNReal a).restrict f, continuous_toFun :=  …
      ha : LE.le 0 a
      x : NNReal
      hx : Membership.mem (quasispectrum NNReal a) x
      ⊢ LT.lt (f x) c
    -/
    exact h x hx
    /-
      🎉 no goals
    -/


lemma nnnorm_cfcₙ_nnreal_lt_iff (f : ℝ≥0 → ℝ≥0) (a : A) (c : ℝ≥0)
    (hf : ContinuousOn f (σₙ ℝ≥0 a) := by cfc_cont_tac) (hf₀ : f 0 = 0 := by cfc_zero_tac)
    (ha : 0 ≤ a := by cfc_tac) : ‖cfcₙ f a‖₊ < c ↔ ∀ x ∈ σₙ ℝ≥0 a, f x < c :=
  ⟨fun h _ hx ↦ apply_le_nnnorm_cfcₙ_nnreal f a hx hf hf₀ ha |>.trans_lt h, nnnorm_cfcₙ_nnreal_lt⟩


