open scoped InnerProductSpace in
lemma IsPositive.spectrumRestricts {f : H →L[𝕜] H} (hf : f.IsPositive) :
    SpectrumRestricts f ContinuousMap.realToNNReal := by
  /-
    𝕜 : Type u_1
    H : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup H
    inst✝³ : InnerProductSpace 𝕜 H
    inst✝² : CompleteSpace H
    inst✝¹ : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
    inst✝ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    hf : f.IsPositive
    ⊢ SpectrumRestricts f ⇑ContinuousMap.realToNNReal
  -/
  rw [SpectrumRestricts.nnreal_iff]
  /-
    𝕜 : Type u_1
    H : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup H
    inst✝³ : InnerProductSpace 𝕜 H
    inst✝² : CompleteSpace H
    inst✝¹ : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
    inst✝ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    hf : f.IsPositive
    ⊢ ∀ (x : Real), Membership.mem (spectrum Real f) x → LE.le 0 x
  -/
  intro c hc
  /-
    𝕜 : Type u_1
    H : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup H
    inst✝³ : InnerProductSpace 𝕜 H
    inst✝² : CompleteSpace H
    inst✝¹ : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
    inst✝ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    hf : f.IsPositive
    c : Real
    hc : Membership.mem (spectrum Real f) c
    ⊢ LE.le 0 c
  -/
  contrapose! hc
  /-
    𝕜 : Type u_1
    H : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup H
    inst✝³ : InnerProductSpace 𝕜 H
    inst✝² : CompleteSpace H
    inst✝¹ : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
    inst✝ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    hf : f.IsPositive
    c : Real
    hc : LT.lt c 0
    ⊢ Not (Membership.mem (spectrum Real f) c)
  -/
  rw [spectrum.not_mem_iff, IsUnit.sub_iff, sub_eq_add_neg, ← map_neg]
  /-
    𝕜 : Type u_1
    H : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup H
    inst✝³ : InnerProductSpace 𝕜 H
    inst✝² : CompleteSpace H
    inst✝¹ : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
    inst✝ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    hf : f.IsPositive
    c : Real
    hc : LT.lt c 0
    ⊢ IsUnit (HAdd.hAdd f ((algebraMap Real (ContinuousLinearMap (RingHom.id 𝕜) H  …
  -/
  rw [← neg_pos] at hc
  /-
    𝕜 : Type u_1
    H : Type u_2
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup H
    inst✝³ : InnerProductSpace 𝕜 H
    inst✝² : CompleteSpace H
    inst✝¹ : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
    inst✝ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    hf : f.IsPositive
    c : Real
    hc : LT.lt 0 (Neg.neg c)
    ⊢ IsUnit (HAdd.hAdd f ((algebraMap Real (ContinuousLinearMap (RingHom.id 𝕜) H  …
  -/
  set c := -c
  exact isUnit_of_forall_le_norm_inner_map _ (c := ⟨c, hc.le⟩) hc fun x ↦ calc
    ‖x‖ ^ 2 * c = re ⟪algebraMap ℝ (H →L[𝕜] H) c x, x⟫_𝕜 := by
      rw [Algebra.algebraMap_eq_smul_one, ← algebraMap_smul 𝕜 c (1 : (H →L[𝕜] H)), coe_smul',
        Pi.smul_apply, one_apply, inner_smul_left, RCLike.algebraMap_eq_ofReal, conj_ofReal,
        re_ofReal_mul, inner_self_eq_norm_sq, mul_comm]
    _ ≤ re ⟪(f + (algebraMap ℝ (H →L[𝕜] H)) c) x, x⟫_𝕜 := by
      simpa only [add_apply, inner_add_left, map_add, le_add_iff_nonneg_left]
        using hf.inner_nonneg_left x
    _ ≤ ‖⟪(f + (algebraMap ℝ (H →L[𝕜] H)) c) x, x⟫_𝕜‖ := RCLike.re_le_norm _


instance : NonnegSpectrumClass ℝ (H →L[𝕜] H) where
  quasispectrum_nonneg_of_nonneg f hf :=
    QuasispectrumRestricts.nnreal_iff.mp <| sub_zero f ▸ hf.spectrumRestricts


/-- Because this takes `ContinuousFunctionalCalculus ℝ IsSelfAdjoint` as an argument, and for
the moment we only have this for `𝕜 := ℂ`, this is not registered as an instance. -/
lemma instStarOrderedRingRCLike
    [ContinuousFunctionalCalculus ℝ (IsSelfAdjoint : (H →L[𝕜] H) → Prop)] :
    StarOrderedRing (H →L[𝕜] H) where
  le_iff f g := by
    /-
      𝕜 : Type u_1
      H : Type u_2
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup H
      inst✝⁴ : InnerProductSpace 𝕜 H
      inst✝³ : CompleteSpace H
      inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
      inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
      inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
      f g : ContinuousLinearMap (RingHom.id 𝕜) H H
      ⊢ Iff (LE.le f g) (Exists fun p => And (Membership.mem (AddSubmonoid.closure ( …
    -/
    constructor
      /-
        case mp
        𝕜 : Type u_1
        H : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
        f g : ContinuousLinearMap (RingHom.id 𝕜) H H
        ⊢ LE.le f g → Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.r …
      -/
    · intro h
      /-
        case mp
        𝕜 : Type u_1
        H : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
        f g : ContinuousLinearMap (RingHom.id 𝕜) H H
        h : LE.le f g
        ⊢ Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s = …
      -/
      rw [le_def] at h
      obtain ⟨p, hp₁, -, hp₃⟩ :=
        CFC.exists_sqrt_of_isSelfAdjoint_of_spectrumRestricts h.1 h.spectrumRestricts
      /-
        case mp.intro.intro.intro
        𝕜 : Type u_1
        H : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
        f g : ContinuousLinearMap (RingHom.id 𝕜) H H
        h : (HSub.hSub g f).IsPositive
        p : ContinuousLinearMap (RingHom.id 𝕜) H H
        hp₁ : IsSelfAdjoint p
        hp₃ : Eq (HPow.hPow p 2) (HSub.hSub g f)
        ⊢ Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s = …
      -/
      refine ⟨p ^ 2, ?_, by symm; rwa [add_comm, ← eq_sub_iff_add_eq]⟩
      /-
        case mp.intro.intro.intro
        𝕜 : Type u_1
        H : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
        f g : ContinuousLinearMap (RingHom.id 𝕜) H H
        h : (HSub.hSub g f).IsPositive
        p : ContinuousLinearMap (RingHom.id 𝕜) H H
        hp₁ : IsSelfAdjoint p
        hp₃ : Eq (HPow.hPow p 2) (HSub.hSub g f)
        ⊢ Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star.sta …
      -/
      exact AddSubmonoid.subset_closure ⟨p, by simp only [hp₁.star_eq, sq]⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        𝕜 : Type u_1
        H : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
        f g : ContinuousLinearMap (RingHom.id 𝕜) H H
        ⊢ (Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s  …
      -/
    · rintro ⟨p, hp, rfl⟩
      /-
        case mpr.intro.intro
        𝕜 : Type u_1
        H : Type u_2
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup H
        inst✝⁴ : InnerProductSpace 𝕜 H
        inst✝³ : CompleteSpace H
        inst✝² : Algebra Real (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝¹ : IsScalarTower Real 𝕜 (ContinuousLinearMap (RingHom.id 𝕜) H H)
        inst✝ : ContinuousFunctionalCalculus Real IsSelfAdjoint
        f p : ContinuousLinearMap (RingHom.id 𝕜) H H
        hp : Membership.mem (AddSubmonoid.closure (Set.range fun s => HMul.hMul (Star. …
        ⊢ LE.le f (HAdd.hAdd f p)
      -/
      rw [le_def, add_sub_cancel_left]
      induction hp using AddSubmonoid.closure_induction with
      | mem _ hf =>
        obtain ⟨f, rfl⟩ := hf
        simpa using ContinuousLinearMap.IsPositive.adjoint_conj isPositive_one f
      | one => exact isPositive_zero
      | mul f g _ _ hf hg => exact hf.add hg


instance instStarOrderedRing {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℂ H] [CompleteSpace H] : StarOrderedRing (H →L[ℂ] H) :=
  instStarOrderedRingRCLike


