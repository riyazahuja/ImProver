theorem RCLike.uniqueContinuousFunctionalCalculus_of_compactSpace_spectrum [TopologicalSpace A]
    [T2Space A] [Ring A] [StarRing A] [Algebra 𝕜 A] [h : ∀ a : A, CompactSpace (spectrum 𝕜 a)] :
    UniqueContinuousFunctionalCalculus 𝕜 A where
  eq_of_continuous_of_map_id s _ φ ψ hφ hψ h :=
    ContinuousMap.starAlgHom_ext_map_X hφ hψ <| by
      /-
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : TopologicalSpace A
        inst✝³ : T2Space A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra 𝕜 A
        h✝ : ∀ (a : A), CompactSpace ↑(spectrum 𝕜 a)
        s : Set 𝕜
        x✝ : CompactSpace ↑s
        φ ψ : StarAlgHom 𝕜 (ContinuousMap (↑s) 𝕜) A
        hφ : Continuous ⇑φ
        hψ : Continuous ⇑ψ
        h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id 𝕜))) (ψ (ContinuousMap.r …
        ⊢ Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (ψ ((Polynomial …
      -/
      convert h using 1
      /-
        case h.e'_2
        𝕜 : Type u_1
        A : Type u_2
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : TopologicalSpace A
        inst✝³ : T2Space A
        inst✝² : Ring A
        inst✝¹ : StarRing A
        inst✝ : Algebra 𝕜 A
        h✝ : ∀ (a : A), CompactSpace ↑(spectrum 𝕜 a)
        s : Set 𝕜
        x✝ : CompactSpace ↑s
        φ ψ : StarAlgHom 𝕜 (ContinuousMap (↑s) 𝕜) A
        hφ : Continuous ⇑φ
        hψ : Continuous ⇑ψ
        h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id 𝕜))) (ψ (ContinuousMap.r …
        ⊢ Eq (φ ((Polynomial.toContinuousMapOnAlgHom s) Polynomial.X)) (φ (ContinuousM …
      -/
      all_goals exact congr_arg _ (by ext; simp)
      /-
        🎉 no goals
      -/
  compactSpace_spectrum := h


instance RCLike.instUniqueContinuousFunctionalCalculus [NormedRing A] [StarRing A]
    [NormedAlgebra 𝕜 A] [CompleteSpace A] : UniqueContinuousFunctionalCalculus 𝕜 A :=
  RCLike.uniqueContinuousFunctionalCalculus_of_compactSpace_spectrum


/-- This map sends `f : C(X, ℝ)` to `Real.toNNReal ∘ f`, bundled as a continuous map `C(X, ℝ≥0)`. -/
noncomputable def toNNReal (f : C(X, ℝ)) : C(X, ℝ≥0) := .realToNNReal |>.comp f


@[fun_prop]
lemma continuous_toNNReal : Continuous (toNNReal (X := X)) := continuous_postcomp _


@[simp]
lemma toNNReal_apply (f : C(X, ℝ)) (x : X) : f.toNNReal x = (f x).toNNReal := rfl


lemma toNNReal_add_add_neg_add_neg_eq (f g : C(X, ℝ)) :
    (f + g).toNNReal + (-f).toNNReal + (-g).toNNReal =
      (-(f + g)).toNNReal + f.toNNReal + g.toNNReal := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    f g : ContinuousMap X Real
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd f g).toNNReal (Neg.neg f).toNNReal) (Neg …
  -/
  ext x
  /-
    case h.a
    X : Type u_1
    inst✝ : TopologicalSpace X
    f g : ContinuousMap X Real
    x : X
    ⊢ Eq ↑((HAdd.hAdd (HAdd.hAdd (HAdd.hAdd f g).toNNReal (Neg.neg f).toNNReal) (N …
  -/
  simp [max_neg_zero, -neg_add_rev]
  /-
    case h.a
    X : Type u_1
    inst✝ : TopologicalSpace X
    f g : ContinuousMap X Real
    x : X
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Max.max (HAdd.hAdd (f x) (g x)) 0) (HAdd.hAdd (Neg …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


lemma toNNReal_mul_add_neg_mul_add_mul_neg_eq (f g : C(X, ℝ)) :
    (f * g).toNNReal + (-f).toNNReal * g.toNNReal + f.toNNReal * (-g).toNNReal =
      (-(f * g)).toNNReal + f.toNNReal * g.toNNReal + (-f).toNNReal * (-g).toNNReal := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    f g : ContinuousMap X Real
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul f g).toNNReal (HMul.hMul (Neg.neg f).toN …
  -/
  ext x
  /-
    case h.a
    X : Type u_1
    inst✝ : TopologicalSpace X
    f g : ContinuousMap X Real
    x : X
    ⊢ Eq ↑((HAdd.hAdd (HAdd.hAdd (HMul.hMul f g).toNNReal (HMul.hMul (Neg.neg f).t …
  -/
  simp [max_neg_zero, add_mul, mul_add]
  /-
    case h.a
    X : Type u_1
    inst✝ : TopologicalSpace X
    f g : ContinuousMap X Real
    x : X
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (Max.max (HMul.hMul (f x) (g x)) 0) (HAdd.hAdd (Neg …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


@[simp]
lemma toNNReal_algebraMap (r : ℝ≥0) :
    (algebraMap ℝ C(X, ℝ) r).toNNReal = algebraMap ℝ≥0 C(X, ℝ≥0) r := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    r : NNReal
    ⊢ Eq ((algebraMap Real (ContinuousMap X Real)) ↑r).toNNReal ((algebraMap NNRea …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
lemma toNNReal_neg_algebraMap (r : ℝ≥0) : (- algebraMap ℝ C(X, ℝ) r).toNNReal = 0 := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    r : NNReal
    ⊢ Eq (Neg.neg ((algebraMap Real (ContinuousMap X Real)) ↑r)).toNNReal 0
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
lemma toNNReal_one : (1 : C(X, ℝ)).toNNReal = 1 := toNNReal_algebraMap 1


@[simp]
lemma toNNReal_neg_one : (-1 : C(X, ℝ)).toNNReal = 0 := toNNReal_neg_algebraMap 1


/-- Given a star `ℝ≥0`-algebra homomorphism `φ` from `C(X, ℝ≥0)` into an `ℝ`-algebra `A`, this is
the unique extension of `φ` from `C(X, ℝ)` to `A` as a star `ℝ`-algebra homomorphism. -/
@[simps]
noncomputable def realContinuousMapOfNNReal (φ : C(X, ℝ≥0) →⋆ₐ[ℝ≥0] A) :
    C(X, ℝ) →⋆ₐ[ℝ] A where
  toFun f := φ f.toNNReal - φ (-f).toNNReal
                 /-
                   X : Type u_1
                   inst✝⁵ : TopologicalSpace X
                   A : Type u_2
                   inst✝⁴ : Ring A
                   inst✝³ : StarRing A
                   inst✝² : Algebra Real A
                   inst✝¹ : TopologicalSpace A
                   inst✝ : TopologicalRing A
                   φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
                   ⊢ Eq ((fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal)) 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
                  /-
                    X : Type u_1
                    inst✝⁵ : TopologicalSpace X
                    A : Type u_2
                    inst✝⁴ : Ring A
                    inst✝³ : StarRing A
                    inst✝² : Algebra Real A
                    inst✝¹ : TopologicalSpace A
                    inst✝ : TopologicalRing A
                    φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
                    ⊢ Eq ((↑{ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), …
                  -/
  map_zero' := by simp
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
    -/
                  /-
                    🎉 no goals
                  -/
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (φ (HAdd.hAdd (HAdd.hAdd (HMul.hMul f g).toNNReal (HMul.hMul (Neg.ne …
      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
    -/
  map_mul' f g := by
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (HAdd.hAdd (HAdd.hAdd (φ (HMul.hMul f g).toNNReal) (HMul.hMul (φ (Ne …
      ⊢ Eq (HSub.hSub (φ (HMul.hMul f g).toNNReal) (φ (Neg.neg (HMul.hMul f g)).toNN …
    -/
    have := congr(φ $(f.toNNReal_mul_add_neg_mul_add_mul_neg_eq g))
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HMul.hMul f g).toNNReal) (HMul. …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HMul.hMul f g).toNNReal) (φ (Neg.neg (HMul.hMul …
    -/
    simp only [map_add, map_mul, sub_mul, mul_sub] at this ⊢
    /-
      case h.e'_2
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HMul.hMul f g).toNNReal) (HMul. …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HMul.hMul f g).toNNReal) (φ (Neg.neg (HMul.hMul …
    -/
    /-
      🎉 no goals
    -/
    rw [← sub_eq_zero] at this ⊢
    /-
      🎉 no goals
    -/
    convert this using 1
    abel
  map_add' f g := by
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      ⊢ Eq ((↑{ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), …
    -/
    have := congr(φ $(f.toNNReal_add_add_neg_add_neg_eq g))
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (φ (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd f g).toNNReal (Neg.neg f).toNNRe …
      ⊢ Eq ((↑{ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), …
    -/
    simp only [map_add] at this ⊢
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (HAdd.hAdd (HAdd.hAdd (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg f).to …
      ⊢ Eq (HSub.hSub (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg (HAdd.hAdd f g)).toNN …
    -/
    rw [← sub_eq_zero] at this ⊢
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HAdd.hAdd f g).toNNReal) (φ (Ne …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg (HAdd.hAdd …
    -/
    convert this using 1
    /-
      case h.e'_2
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      f g : ContinuousMap X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HAdd.hAdd f g).toNNReal) (φ (Ne …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg (HAdd.hAdd …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  commutes' r := by
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      r : Real
      ⊢ Eq ((↑↑{ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal) …
    -/
    simp only
    /-
      X : Type u_1
      inst✝⁵ : TopologicalSpace X
      A : Type u_2
      inst✝⁴ : Ring A
      inst✝³ : StarRing A
      inst✝² : Algebra Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
      r : Real
      ⊢ Eq (HSub.hSub (φ ((algebraMap Real (ContinuousMap X Real)) r).toNNReal) (φ ( …
    -/
    obtain (hr | hr) := le_total 0 r
      /-
        case inl
        X : Type u_1
        inst✝⁵ : TopologicalSpace X
        A : Type u_2
        inst✝⁴ : Ring A
        inst✝³ : StarRing A
        inst✝² : Algebra Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
        r : Real
        hr : LE.le 0 r
        ⊢ Eq (HSub.hSub (φ ((algebraMap Real (ContinuousMap X Real)) r).toNNReal) (φ ( …
      -/
    · lift r to ℝ≥0 using hr
      simpa only [ContinuousMap.toNNReal_algebraMap, ContinuousMap.toNNReal_neg_algebraMap,
        map_zero, sub_zero] using AlgHomClass.commutes φ r
      /-
        case inr
        X : Type u_1
        inst✝⁵ : TopologicalSpace X
        A : Type u_2
        inst✝⁴ : Ring A
        inst✝³ : StarRing A
        inst✝² : Algebra Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
        r : Real
        hr : LE.le r 0
        ⊢ Eq (HSub.hSub (φ ((algebraMap Real (ContinuousMap X Real)) r).toNNReal) (φ ( …
      -/
    · rw [← neg_neg r, ← map_neg, neg_neg (-r)]
      /-
        case inr
        X : Type u_1
        inst✝⁵ : TopologicalSpace X
        A : Type u_2
        inst✝⁴ : Ring A
        inst✝³ : StarRing A
        inst✝² : Algebra Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
        r : Real
        hr : LE.le r 0
        ⊢ Eq (HSub.hSub (φ ((algebraMap Real (ContinuousMap X Real)) (Neg.neg (Neg.neg …
      -/
      rw [← neg_nonneg] at hr
      /-
        case inr
        X : Type u_1
        inst✝⁵ : TopologicalSpace X
        A : Type u_2
        inst✝⁴ : Ring A
        inst✝³ : StarRing A
        inst✝² : Algebra Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
        r : Real
        hr : LE.le 0 (Neg.neg r)
        ⊢ Eq (HSub.hSub (φ ((algebraMap Real (ContinuousMap X Real)) (Neg.neg (Neg.neg …
      -/
      lift -r to ℝ≥0 using hr with r
      simpa only [map_neg, ContinuousMap.toNNReal_neg_algebraMap, map_zero,
        ContinuousMap.toNNReal_algebraMap, zero_sub, neg_inj] using AlgHomClass.commutes φ r
                    /-
                      X : Type u_1
                      inst✝⁵ : TopologicalSpace X
                      A : Type u_2
                      inst✝⁴ : Ring A
                      inst✝³ : StarRing A
                      inst✝² : Algebra Real A
                      inst✝¹ : TopologicalSpace A
                      inst✝ : TopologicalRing A
                      φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
                      f : ContinuousMap X Real
                      ⊢ Eq ((↑↑{ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal) …
                    -/
  map_star' f := by simp only [star_trivial, star_sub, ← map_star]
                    /-
                      🎉 no goals
                    -/


@[fun_prop]
lemma continuous_realContinuousMapOfNNReal (φ : C(X, ℝ≥0) →⋆ₐ[ℝ≥0] A)
    (hφ : Continuous φ) : Continuous φ.realContinuousMapOfNNReal := by
  /-
    X : Type u_1
    inst✝⁵ : TopologicalSpace X
    A : Type u_2
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra Real A
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalRing A
    φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    hφ : Continuous ⇑φ
    ⊢ Continuous ⇑φ.realContinuousMapOfNNReal
  -/
  simp [realContinuousMapOfNNReal]
  /-
    X : Type u_1
    inst✝⁵ : TopologicalSpace X
    A : Type u_2
    inst✝⁴ : Ring A
    inst✝³ : StarRing A
    inst✝² : Algebra Real A
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalRing A
    φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    hφ : Continuous ⇑φ
    ⊢ Continuous fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[simp high]
lemma realContinuousMapOfNNReal_apply_comp_toReal (φ : C(X, ℝ≥0) →⋆ₐ[ℝ≥0] A)
    (f : C(X, ℝ≥0)) :
    φ.realContinuousMapOfNNReal ((ContinuousMap.mk toReal continuous_coe).comp f) = φ f := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    A : Type u_2
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra Real A
    φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    f : ContinuousMap X NNReal
    ⊢ Eq (φ.realContinuousMapOfNNReal ({ toFun := NNReal.toReal, continuous_toFun  …
  -/
  simp only [realContinuousMapOfNNReal_apply]
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    A : Type u_2
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra Real A
    φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    f : ContinuousMap X NNReal
    ⊢ Eq (HSub.hSub (φ ({ toFun := NNReal.toReal, continuous_toFun := NNReal.conti …
  -/
  convert_to φ f - φ 0 = φ f using 2
  /-
    case h.e'_2.h.e'_5
    X : Type u_1
    inst✝³ : TopologicalSpace X
    A : Type u_2
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra Real A
    φ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    f : ContinuousMap X NNReal
    ⊢ Eq (φ ({ toFun := NNReal.toReal, continuous_toFun := NNReal.continuous_coe } …
  -/
  on_goal -1 => rw [map_zero, sub_zero]
  all_goals
    congr
    ext x
    simp


lemma realContinuousMapOfNNReal_injective :
    Function.Injective (realContinuousMapOfNNReal (X := X) (A := A)) := by
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    A : Type u_2
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra Real A
    ⊢ Function.Injective StarAlgHom.realContinuousMapOfNNReal
  -/
  intro φ ψ h
  /-
    X : Type u_1
    inst✝³ : TopologicalSpace X
    A : Type u_2
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra Real A
    φ ψ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    h : Eq φ.realContinuousMapOfNNReal ψ.realContinuousMapOfNNReal
    ⊢ Eq φ ψ
  -/
  ext f
  /-
    case h
    X : Type u_1
    inst✝³ : TopologicalSpace X
    A : Type u_2
    inst✝² : Ring A
    inst✝¹ : StarRing A
    inst✝ : Algebra Real A
    φ ψ : StarAlgHom NNReal (ContinuousMap X NNReal) A
    h : Eq φ.realContinuousMapOfNNReal ψ.realContinuousMapOfNNReal
    f : ContinuousMap X NNReal
    ⊢ Eq (φ f) (ψ f)
  -/
  simpa using congr($(h) ((ContinuousMap.mk toReal continuous_coe).comp f))
  /-
    🎉 no goals
  -/


instance NNReal.instUniqueContinuousFunctionalCalculus [UniqueContinuousFunctionalCalculus ℝ A] :
    UniqueContinuousFunctionalCalculus ℝ≥0 A where
  compactSpace_spectrum a := by
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      a : A
      ⊢ CompactSpace ↑(spectrum NNReal a)
    -/
    have : CompactSpace (spectrum ℝ a) := UniqueContinuousFunctionalCalculus.compactSpace_spectrum a
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      a : A
      this : CompactSpace ↑(spectrum Real a)
      ⊢ CompactSpace ↑(spectrum NNReal a)
    -/
    rw [← isCompact_iff_compactSpace] at *
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      a : A
      this : IsCompact (spectrum Real a)
      ⊢ IsCompact (spectrum NNReal a)
    -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      ⊢ Eq φ ψ
    -/
    rw [← spectrum.preimage_algebraMap ℝ]
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      a : A
      this : IsCompact (spectrum Real a)
      ⊢ IsCompact (Set.preimage (⇑(algebraMap NNReal Real)) (spectrum Real a))
    -/
    exact isClosed_nonneg.isClosedEmbedding_subtypeVal.isCompact_preimage <| by assumption
    /-
      🎉 no goals
    -/
  eq_of_continuous_of_map_id s hs φ ψ hφ hψ h := by
    let s' : Set ℝ := (↑) '' s
    let e : s ≃ₜ s' :=
      { toFun := Subtype.map (↑) (by simp [s'])
        invFun := Subtype.map Real.toNNReal (by simp [s'])
        left_inv := fun _ ↦ by ext; simp
        right_inv := fun x ↦ by
          ext
          obtain ⟨y, -, hy⟩ := x.2
          simpa using hy ▸ NNReal.coe_nonneg y
        continuous_toFun := continuous_coe.subtype_map (by simp [s'])
        continuous_invFun := continuous_real_toNNReal.subtype_map (by simp [s']) }
    have (ξ : C(s, ℝ≥0) →⋆ₐ[ℝ≥0] A) (hξ : Continuous ξ) :
        (let ξ' := ξ.realContinuousMapOfNNReal.comp <| ContinuousMap.compStarAlgHom' ℝ ℝ e
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      this :
        ∀ (ξ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMap.restrict s' (ContinuousMap.i …
      ⊢ Eq φ ψ
    -/
        Continuous ξ' ∧ ξ' (.restrict s' <| .id ℝ) = ξ (.restrict s <| .id ℝ≥0)) := by
    /-
      case intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      this :
        ∀ (ξ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMap.restrict s' (ContinuousMap.i …
      hφ' : Continuous ⇑(φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hφ_id : Eq ((φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      ⊢ Eq φ ψ
    -/
      intro ξ'
    /-
      case intro.intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      this :
        ∀ (ξ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMap.restrict s' (ContinuousMap.i …
      hφ' : Continuous ⇑(φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hφ_id : Eq ((φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hψ' : Continuous ⇑(ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hψ_id : Eq ((ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      ⊢ Eq φ ψ
    -/
      refine ⟨ξ.continuous_realContinuousMapOfNNReal hξ |>.comp <|
        ContinuousMap.continuous_precomp _, ?_⟩
      exact ξ.realContinuousMapOfNNReal_apply_comp_toReal (.restrict s <| .id ℝ≥0)
    /-
      case intro.intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      this :
        ∀ (ξ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMap.restrict s' (ContinuousMap.i …
      hφ' : Continuous ⇑(φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hφ_id : Eq ((φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hψ' : Continuous ⇑(ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hψ_id : Eq ((ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hs' : CompactSpace ↑s'
      h' : Eq (φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' Real  …
      ⊢ Eq φ ψ
    -/
    obtain ⟨hφ', hφ_id⟩ := this φ hφ
    obtain ⟨hψ', hψ_id⟩ := this ψ hψ
    have hs' : CompactSpace s' := e.compactSpace
    have h' := UniqueContinuousFunctionalCalculus.eq_of_continuous_of_map_id s' _ _ hφ' hψ'
    /-
      case intro.intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      this✝ :
        ∀ (ξ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMap.restrict s' (ContinuousMap.i …
      hφ' : Continuous ⇑(φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hφ_id : Eq ((φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hψ' : Continuous ⇑(ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hψ_id : Eq ((ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hs' : CompactSpace ↑s'
      h' : Eq (φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' Real  …
      h'' : Eq ((φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' Rea …
      this : Eq ((ContinuousMap.compStarAlgHom' Real Real ↑e).comp (ContinuousMap.co …
      ⊢ Eq φ ψ
    -/
      (hφ_id ▸ hψ_id ▸ h)
    /-
      case intro.intro
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      A : Type u_2
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra Real A
      inst✝² : TopologicalSpace A
      inst✝¹ : TopologicalRing A
      inst✝ : UniqueContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      φ ψ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ (ContinuousMap.restrict s (ContinuousMap.id NNReal))) (ψ (Continuous …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      this✝ :
        ∀ (ξ : StarAlgHom NNReal (ContinuousMap (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMap.restrict s' (ContinuousMap.i …
      hφ' : Continuous ⇑(φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hφ_id : Eq ((φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hψ' : Continuous ⇑(ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlg …
      hψ_id : Eq ((ψ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' R …
      hs' : CompactSpace ↑s'
      h' : Eq (φ.realContinuousMapOfNNReal.comp (ContinuousMap.compStarAlgHom' Real  …
      this : Eq ((ContinuousMap.compStarAlgHom' Real Real ↑e).comp (ContinuousMap.co …
      h'' : Eq φ.realContinuousMapOfNNReal ψ.realContinuousMapOfNNReal
      ⊢ Eq φ ψ
    -/
    have h'' := congr($(h').comp <| ContinuousMap.compStarAlgHom' ℝ ℝ (e.symm : C(s', s)))
    /-
      🎉 no goals
    -/
    have : (ContinuousMap.compStarAlgHom' ℝ ℝ (e : C(s, s'))).comp
        (ContinuousMap.compStarAlgHom' ℝ ℝ (e.symm : C(s', s))) = StarAlgHom.id _ _ := by
      ext1; simp
    simp only [StarAlgHom.comp_assoc, this, StarAlgHom.comp_id] at h''
    exact StarAlgHom.realContinuousMapOfNNReal_injective h''


open NonUnitalStarAlgebra in
theorem RCLike.uniqueNonUnitalContinuousFunctionalCalculus_of_compactSpace_quasispectrum
    [TopologicalSpace A] [T2Space A] [NonUnitalRing A] [StarRing A] [Module 𝕜 A]
    [IsScalarTower 𝕜 A A] [SMulCommClass 𝕜 A A] [h : ∀ a : A, CompactSpace (quasispectrum 𝕜 a)] :
    UniqueNonUnitalContinuousFunctionalCalculus 𝕜 A where
  eq_of_continuous_of_map_id s hs _inst h0 φ ψ hφ hψ h := by
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : T2Space A
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      h✝ : ∀ (a : A), CompactSpace ↑(quasispectrum 𝕜 a)
      s : Set 𝕜
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑s) 𝕜) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id 𝕜),  …
      ⊢ Eq φ ψ
    -/
    rw [DFunLike.ext'_iff, ← Set.eqOn_univ, ← (ContinuousMapZero.adjoin_id_dense h0).closure_eq]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : T2Space A
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      h✝ : ∀ (a : A), CompactSpace ↑(quasispectrum 𝕜 a)
      s : Set 𝕜
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑s) 𝕜) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id 𝕜),  …
      ⊢ Set.EqOn (⇑φ) (⇑ψ) (closure ↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singl …
    -/
    refine Set.EqOn.closure (fun f hf ↦ ?_) hφ hψ
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : T2Space A
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      h✝ : ∀ (a : A), CompactSpace ↑(quasispectrum 𝕜 a)
      s : Set 𝕜
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑s) 𝕜) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id 𝕜),  …
      f : ContinuousMapZero (↑s) 𝕜
      hf : Membership.mem (↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Con …
      ⊢ Eq (φ f) (ψ f)
    -/
    rw [← NonUnitalStarAlgHom.mem_equalizer]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : T2Space A
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      h✝ : ∀ (a : A), CompactSpace ↑(quasispectrum 𝕜 a)
      s : Set 𝕜
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑s) 𝕜) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id 𝕜),  …
      f : ContinuousMapZero (↑s) 𝕜
      hf : Membership.mem (↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Con …
      ⊢ Membership.mem (NonUnitalStarAlgHom.equalizer φ ψ) f
    -/
    apply adjoin_le ?_ hf
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : T2Space A
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      h✝ : ∀ (a : A), CompactSpace ↑(quasispectrum 𝕜 a)
      s : Set 𝕜
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑s) 𝕜) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id 𝕜),  …
      f : ContinuousMapZero (↑s) 𝕜
      hf : Membership.mem (↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Con …
      ⊢ HasSubset.Subset (Singleton.singleton (ContinuousMapZero.id h0)) ↑(NonUnital …
    -/
    rw [Set.singleton_subset_iff]
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝⁷ : RCLike 𝕜
      inst✝⁶ : TopologicalSpace A
      inst✝⁵ : T2Space A
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module 𝕜 A
      inst✝¹ : IsScalarTower 𝕜 A A
      inst✝ : SMulCommClass 𝕜 A A
      h✝ : ∀ (a : A), CompactSpace ↑(quasispectrum 𝕜 a)
      s : Set 𝕜
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom 𝕜 (ContinuousMapZero (↑s) 𝕜) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id 𝕜),  …
      f : ContinuousMapZero (↑s) 𝕜
      hf : Membership.mem (↑(NonUnitalStarAlgebra.adjoin 𝕜 (Singleton.singleton (Con …
      ⊢ Membership.mem (↑(NonUnitalStarAlgHom.equalizer φ ψ)) (ContinuousMapZero.id  …
    -/
    exact h
    /-
      🎉 no goals
    -/
  compactSpace_quasispectrum := h


instance RCLike.instUniqueNonUnitalContinuousFunctionalCalculus [NonUnitalNormedRing A]
    [StarRing A] [CompleteSpace A] [NormedSpace 𝕜 A] [IsScalarTower 𝕜 A A] [SMulCommClass 𝕜 A A] :
    UniqueNonUnitalContinuousFunctionalCalculus 𝕜 A :=
  RCLike.uniqueNonUnitalContinuousFunctionalCalculus_of_compactSpace_quasispectrum


/-- This map sends `f : C(X, ℝ)` to `Real.toNNReal ∘ f`, bundled as a continuous map `C(X, ℝ≥0)`. -/
                                                                                       /-
                                                                                         X : Type u_1
                                                                                         inst✝¹ : TopologicalSpace X
                                                                                         inst✝ : Zero X
                                                                                         f : ContinuousMapZero X Real
                                                                                         ⊢ Eq ((ContinuousMap.realToNNReal.comp ↑f) 0) 0
                                                                                       -/
noncomputable def toNNReal (f : C(X, ℝ)₀) : C(X, ℝ≥0)₀ := ⟨.realToNNReal |>.comp f, by simp⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
lemma toNNReal_apply (f : C(X, ℝ)₀) (x : X) : f.toNNReal x = Real.toNNReal (f x) := rfl


@[fun_prop]
lemma continuous_toNNReal : Continuous (toNNReal (X := X)) := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    ⊢ Continuous ContinuousMapZero.toNNReal
  -/
  rw [continuous_induced_rng]
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    ⊢ Continuous (Function.comp _root_.toContinuousMap ContinuousMapZero.toNNReal)
  -/
  convert_to Continuous (ContinuousMap.toNNReal ∘ ((↑) : C(X, ℝ)₀ → C(X, ℝ))) using 1
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    ⊢ Continuous (Function.comp ContinuousMap.toNNReal _root_.toContinuousMap)
  -/
  exact ContinuousMap.continuous_postcomp _ |>.comp continuous_induced_dom
  /-
    🎉 no goals
  -/


lemma toContinuousMapHom_toNNReal (f : C(X, ℝ)₀) :
    (toContinuousMapHom (X := X) (R := ℝ) f).toNNReal =
      toContinuousMapHom (X := X) (R := ℝ≥0) f.toNNReal :=
  rfl


@[simp]
lemma toNNReal_smul (r : ℝ≥0) (f : C(X, ℝ)₀) : (r • f).toNNReal = r • f.toNNReal := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    r : NNReal
    f : ContinuousMapZero X Real
    ⊢ Eq (HSMul.hSMul r f).toNNReal (HSMul.hSMul r f.toNNReal)
  -/
  ext x
  /-
    case h.a
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    r : NNReal
    f : ContinuousMapZero X Real
    x : X
    ⊢ Eq ↑((HSMul.hSMul r f).toNNReal x) ↑((HSMul.hSMul r f.toNNReal) x)
  -/
  by_cases h : 0 ≤ f x
    /-
      case pos
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : Zero X
      r : NNReal
      f : ContinuousMapZero X Real
      x : X
      h : LE.le 0 (f x)
      ⊢ Eq ↑((HSMul.hSMul r f).toNNReal x) ↑((HSMul.hSMul r f.toNNReal) x)
    -/
  · simpa [max_eq_left h, NNReal.smul_def] using mul_nonneg r.coe_nonneg h
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : Zero X
      r : NNReal
      f : ContinuousMapZero X Real
      x : X
      h : Not (LE.le 0 (f x))
      ⊢ Eq ↑((HSMul.hSMul r f).toNNReal x) ↑((HSMul.hSMul r f.toNNReal) x)
    -/
  · push_neg at h
    simpa [max_eq_right h.le, NNReal.smul_def]
      using mul_nonpos_of_nonneg_of_nonpos r.coe_nonneg h.le


@[simp]
lemma toNNReal_neg_smul (r : ℝ≥0) (f : C(X, ℝ)₀) : (-(r • f)).toNNReal = r • (-f).toNNReal := by
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    r : NNReal
    f : ContinuousMapZero X Real
    ⊢ Eq (Neg.neg (HSMul.hSMul r f)).toNNReal (HSMul.hSMul r (Neg.neg f).toNNReal)
  -/
  rw [NNReal.smul_def, ← smul_neg, ← NNReal.smul_def, toNNReal_smul]
  /-
    🎉 no goals
  -/


lemma toNNReal_mul_add_neg_mul_add_mul_neg_eq (f g : C(X, ℝ)₀) :
    ((f * g).toNNReal + (-f).toNNReal * g.toNNReal + f.toNNReal * (-g).toNNReal) =
    ((-(f * g)).toNNReal + f.toNNReal * g.toNNReal + (-f).toNNReal * (-g).toNNReal) := by
  -- Without this, Lean fails to find the instance in time
  have : SemilinearMapClass (C(X, ℝ≥0)₀ →⋆ₙₐ[ℝ≥0] C(X, ℝ≥0)) (RingHom.id ℝ≥0)
    C(X, ℝ≥0)₀ C(X, ℝ≥0) := NonUnitalAlgHomClass.instLinearMapClass
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    f g : ContinuousMapZero X Real
    this : SemilinearMapClass (NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNR …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul f g).toNNReal (HMul.hMul (Neg.neg f).toN …
  -/
  apply toContinuousMap_injective
  simpa only [← toContinuousMapHom_apply, map_add, map_mul, map_neg, toContinuousMapHom_toNNReal]
    using (f : C(X, ℝ)).toNNReal_mul_add_neg_mul_add_mul_neg_eq g


lemma toNNReal_add_add_neg_add_neg_eq (f g : C(X, ℝ)₀) :
    ((f + g).toNNReal + (-f).toNNReal + (-g).toNNReal) =
      ((-(f + g)).toNNReal + f.toNNReal + g.toNNReal) := by
  -- Without this, Lean fails to find the instance in time
  have : SemilinearMapClass (C(X, ℝ≥0)₀ →⋆ₙₐ[ℝ≥0] C(X, ℝ≥0)) (RingHom.id ℝ≥0)
    C(X, ℝ≥0)₀ C(X, ℝ≥0) := NonUnitalAlgHomClass.instLinearMapClass
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : Zero X
    f g : ContinuousMapZero X Real
    this : SemilinearMapClass (NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNR …
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd f g).toNNReal (Neg.neg f).toNNReal) (Neg …
  -/
  apply toContinuousMap_injective
  simpa only [← toContinuousMapHom_apply, map_add, map_mul, map_neg, toContinuousMapHom_toNNReal]
    using (f : C(X, ℝ)).toNNReal_add_add_neg_add_neg_eq g


/-- Given a non-unital star `ℝ≥0`-algebra homomorphism `φ` from `C(X, ℝ≥0)₀` into a non-unital
`ℝ`-algebra `A`, this is the unique extension of `φ` from `C(X, ℝ)₀` to `A` as a non-unital
star `ℝ`-algebra homomorphism. -/
@[simps]
noncomputable def realContinuousMapZeroOfNNReal (φ : C(X, ℝ≥0)₀ →⋆ₙₐ[ℝ≥0] A) :
    C(X, ℝ)₀ →⋆ₙₐ[ℝ] A where
  toFun f := φ f.toNNReal - φ (-f).toNNReal
                  /-
                    X : Type u_1
                    inst✝⁶ : TopologicalSpace X
                    inst✝⁵ : Zero X
                    A : Type u_2
                    inst✝⁴ : NonUnitalRing A
                    inst✝³ : StarRing A
                    inst✝² : Module Real A
                    inst✝¹ : TopologicalSpace A
                    inst✝ : TopologicalRing A
                    φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
                    ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
  map_mul' f g := by
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
    -/
    have := congr(φ $(f.toNNReal_mul_add_neg_mul_add_mul_neg_eq g))
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (φ (HAdd.hAdd (HAdd.hAdd (HMul.hMul f g).toNNReal (HMul.hMul (Neg.ne …
      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
    -/
    simp only [map_add, map_mul, sub_mul, mul_sub] at this ⊢
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (HAdd.hAdd (HAdd.hAdd (φ (HMul.hMul f g).toNNReal) (HMul.hMul (φ (Ne …
      ⊢ Eq (HSub.hSub (φ (HMul.hMul f g).toNNReal) (φ (Neg.neg (HMul.hMul f g)).toNN …
    -/
    rw [← sub_eq_zero] at this ⊢
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
    -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HMul.hMul f g).toNNReal) (HMul. …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HMul.hMul f g).toNNReal) (φ (Neg.neg (HMul.hMul …
    -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (φ (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd f g).toNNReal (Neg.neg f).toNNRe …
      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
    -/
    rw [← this]
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      r : Real
      f : ContinuousMapZero X Real
      ⊢ Eq ((fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal)) (HSMul.hSMu …
    -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (HAdd.hAdd (HAdd.hAdd (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg f).to …
      ⊢ Eq (HSub.hSub (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg (HAdd.hAdd f g)).toNN …
    -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      r : Real
      f : ContinuousMapZero X Real
      ⊢ Eq (HSub.hSub (φ (HSMul.hSMul r f).toNNReal) (φ (Neg.neg (HSMul.hSMul r f)). …
    -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HMul.hMul f g).toNNReal) (HMul. …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HMul.hMul f g).toNNReal) (φ (Neg.neg (HMul.hMul …
    -/
      /-
        case pos
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r : Real
        f : ContinuousMapZero X Real
        hr : LE.le 0 r
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul r f).toNNReal) (φ (Neg.neg (HSMul.hSMul r f)). …
      -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HAdd.hAdd f g).toNNReal) (φ (Ne …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg (HAdd.hAdd …
    -/
      /-
        case pos.intro
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        f : ContinuousMapZero X Real
        r : NNReal
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul (↑r) f).toNNReal) (φ (Neg.neg (HSMul.hSMul (↑r …
      -/
    /-
      🎉 no goals
    -/
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r : Real
        f : ContinuousMapZero X Real
        hr : Not (LE.le 0 r)
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul r f).toNNReal) (φ (Neg.neg (HSMul.hSMul r f)). …
      -/
    /-
      X : Type u_1
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : Zero X
      A : Type u_2
      inst✝⁴ : NonUnitalRing A
      inst✝³ : StarRing A
      inst✝² : Module Real A
      inst✝¹ : TopologicalSpace A
      inst✝ : TopologicalRing A
      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
      f g : ContinuousMapZero X Real
      this : Eq (HSub.hSub (HAdd.hAdd (HAdd.hAdd (φ (HAdd.hAdd f g).toNNReal) (φ (Ne …
      ⊢ Eq (HSub.hSub (HSub.hSub (φ (HAdd.hAdd f g).toNNReal) (φ (Neg.neg (HAdd.hAdd …
    -/
      /-
        case neg
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r : Real
        f : ContinuousMapZero X Real
        hr : LT.lt 0 (Neg.neg r)
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul r f).toNNReal) (φ (Neg.neg (HSMul.hSMul r f)). …
      -/
    /-
      🎉 no goals
    -/
      /-
        case neg
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r : Real
        f : ContinuousMapZero X Real
        hr : LT.lt 0 (Neg.neg r)
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul r f).toNNReal) (φ (HSMul.hSMul (Neg.neg r) f). …
      -/
    abel
      /-
        case neg
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r : Real
        f : ContinuousMapZero X Real
        hr : LT.lt 0 (Neg.neg r)
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul (Neg.neg (Neg.neg r)) f).toNNReal) (φ (HSMul.h …
      -/
    /-
      🎉 no goals
    -/
      /-
        case neg
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r : Real
        f : ContinuousMapZero X Real
        hr : LT.lt 0 (Neg.neg r)
        ⊢ Eq (HSub.hSub (φ (HSMul.hSMul (Neg.neg (Neg.neg r)) f).toNNReal) (φ (HSMul.h …
      -/
    /-
      🎉 no goals
    -/
  map_add' f g := by
    have := congr(φ $(f.toNNReal_add_add_neg_add_neg_eq g))
      /-
        case neg.intro
        X : Type u_1
        inst✝⁶ : TopologicalSpace X
        inst✝⁵ : Zero X
        A : Type u_2
        inst✝⁴ : NonUnitalRing A
        inst✝³ : StarRing A
        inst✝² : Module Real A
        inst✝¹ : TopologicalSpace A
        inst✝ : TopologicalRing A
        φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
        r✝ : Real
        f : ContinuousMapZero X Real
        r : NNReal
        hr✝ hr : LT.lt 0 ↑r
        ⊢ Eq (HSub.hSub (HSMul.hSMul r (φ (Neg.neg f).toNNReal)) (HSMul.hSMul r (φ f.t …
      -/
    simp only [map_add, map_mul, sub_mul, mul_sub] at this ⊢
      /-
        🎉 no goals
      -/
    rw [← sub_eq_zero] at this ⊢
    rw [← this]
    abel
  map_smul' r f := by
    simp only [MonoidHom.id_apply]
    by_cases hr : 0 ≤ r
    · lift r to ℝ≥0 using hr
      simp only [← smul_def, toNNReal_smul, map_smul, toNNReal_neg_smul, smul_sub]
    · rw [not_le, ← neg_pos] at hr
      rw [← neg_smul]
      nth_rw 1 [← neg_neg r]
      nth_rw 3 [← neg_neg r]
      lift -r to ℝ≥0 using hr.le with r
      simp only [neg_smul, ← smul_def, toNNReal_neg_smul, map_smul, toNNReal_smul, smul_sub,
        sub_neg_eq_add]
      rw [sub_eq_add_neg, add_comm]
                    /-
                      X : Type u_1
                      inst✝⁶ : TopologicalSpace X
                      inst✝⁵ : Zero X
                      A : Type u_2
                      inst✝⁴ : NonUnitalRing A
                      inst✝³ : StarRing A
                      inst✝² : Module Real A
                      inst✝¹ : TopologicalSpace A
                      inst✝ : TopologicalRing A
                      φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
                      f : ContinuousMapZero X Real
                      ⊢ Eq ({ toFun := fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal), m …
                    -/
  map_star' f := by simp only [star_trivial, star_sub, ← map_star]
                    /-
                      🎉 no goals
                    -/


@[fun_prop]
lemma continuous_realContinuousMapZeroOfNNReal (φ : C(X, ℝ≥0)₀ →⋆ₙₐ[ℝ≥0] A)
    (hφ : Continuous φ) : Continuous φ.realContinuousMapZeroOfNNReal := by
  /-
    X : Type u_1
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : Zero X
    A : Type u_2
    inst✝⁴ : NonUnitalRing A
    inst✝³ : StarRing A
    inst✝² : Module Real A
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalRing A
    φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    hφ : Continuous ⇑φ
    ⊢ Continuous ⇑φ.realContinuousMapZeroOfNNReal
  -/
  simp [realContinuousMapZeroOfNNReal]
  /-
    X : Type u_1
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : Zero X
    A : Type u_2
    inst✝⁴ : NonUnitalRing A
    inst✝³ : StarRing A
    inst✝² : Module Real A
    inst✝¹ : TopologicalSpace A
    inst✝ : TopologicalRing A
    φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    hφ : Continuous ⇑φ
    ⊢ Continuous fun f => HSub.hSub (φ f.toNNReal) (φ (Neg.neg f).toNNReal)
  -/
  fun_prop
  /-
    🎉 no goals
  -/


@[simp high]
lemma realContinuousMapZeroOfNNReal_apply_comp_toReal (φ : C(X, ℝ≥0)₀ →⋆ₙₐ[ℝ≥0] A)
    (f : C(X, ℝ≥0)₀) :
    φ.realContinuousMapZeroOfNNReal ((ContinuousMapZero.mk ⟨toReal, continuous_coe⟩ rfl).comp f) =
      φ f := by
  /-
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Zero X
    A : Type u_2
    inst✝² : NonUnitalRing A
    inst✝¹ : StarRing A
    inst✝ : Module Real A
    φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    f : ContinuousMapZero X NNReal
    ⊢ Eq (φ.realContinuousMapZeroOfNNReal ({ toFun := NNReal.toReal, continuous_to …
  -/
  simp only [realContinuousMapZeroOfNNReal_apply]
  /-
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Zero X
    A : Type u_2
    inst✝² : NonUnitalRing A
    inst✝¹ : StarRing A
    inst✝ : Module Real A
    φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    f : ContinuousMapZero X NNReal
    ⊢ Eq (HSub.hSub (φ ({ toFun := NNReal.toReal, continuous_toFun := NNReal.conti …
  -/
  convert_to φ f - φ 0 = φ f using 2
  /-
    case h.e'_2.h.e'_5
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Zero X
    A : Type u_2
    inst✝² : NonUnitalRing A
    inst✝¹ : StarRing A
    inst✝ : Module Real A
    φ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    f : ContinuousMapZero X NNReal
    ⊢ Eq (φ ({ toFun := NNReal.toReal, continuous_toFun := NNReal.continuous_coe,  …
  -/
  on_goal -1 => rw [map_zero, sub_zero]
  all_goals
    congr
    ext x
    simp


lemma realContinuousMapZeroOfNNReal_injective :
    Function.Injective (realContinuousMapZeroOfNNReal (X := X) (A := A)) := by
  /-
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Zero X
    A : Type u_2
    inst✝² : NonUnitalRing A
    inst✝¹ : StarRing A
    inst✝ : Module Real A
    ⊢ Function.Injective NonUnitalStarAlgHom.realContinuousMapZeroOfNNReal
  -/
  intro φ ψ h
  /-
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Zero X
    A : Type u_2
    inst✝² : NonUnitalRing A
    inst✝¹ : StarRing A
    inst✝ : Module Real A
    φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    h : Eq φ.realContinuousMapZeroOfNNReal ψ.realContinuousMapZeroOfNNReal
    ⊢ Eq φ ψ
  -/
  ext f
  /-
    case h
    X : Type u_1
    inst✝⁴ : TopologicalSpace X
    inst✝³ : Zero X
    A : Type u_2
    inst✝² : NonUnitalRing A
    inst✝¹ : StarRing A
    inst✝ : Module Real A
    φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero X NNReal) A
    h : Eq φ.realContinuousMapZeroOfNNReal ψ.realContinuousMapZeroOfNNReal
    f : ContinuousMapZero X NNReal
    ⊢ Eq (φ f) (ψ f)
  -/
  simpa using congr($(h) ((ContinuousMapZero.mk ⟨toReal, continuous_coe⟩ rfl).comp f))
  /-
    🎉 no goals
  -/


instance NNReal.instUniqueNonUnitalContinuousFunctionalCalculus
    [TopologicalSpace A] [TopologicalRing A] [IsScalarTower ℝ A A] [SMulCommClass ℝ A A]
    [UniqueNonUnitalContinuousFunctionalCalculus ℝ A] :
    UniqueNonUnitalContinuousFunctionalCalculus ℝ≥0 A where
  compactSpace_quasispectrum a := by
    have : CompactSpace (quasispectrum ℝ a) :=
      UniqueNonUnitalContinuousFunctionalCalculus.compactSpace_quasispectrum a
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      a : A
      this : CompactSpace ↑(quasispectrum Real a)
      ⊢ CompactSpace ↑(quasispectrum NNReal a)
    -/
    rw [← isCompact_iff_compactSpace] at *
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      a : A
      this : IsCompact (quasispectrum Real a)
      ⊢ IsCompact (quasispectrum NNReal a)
    -/
    rw [← quasispectrum.preimage_algebraMap ℝ]
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      ⊢ Eq φ ψ
    -/
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      a : A
      this : IsCompact (quasispectrum Real a)
      ⊢ IsCompact (Set.preimage (⇑(algebraMap NNReal Real)) (quasispectrum Real a))
    -/
    exact isClosed_nonneg.isClosedEmbedding_subtypeVal.isCompact_preimage <| by assumption
    /-
      🎉 no goals
    -/
  eq_of_continuous_of_map_id s hs _inst h0 φ ψ hφ hψ h := by
    let s' : Set ℝ := (↑) '' s
    let e : s ≃ₜ s' :=
      { toFun := Subtype.map (↑) (by simp [s'])
        invFun := Subtype.map Real.toNNReal (by simp [s'])
        left_inv := fun _ ↦ by ext; simp
        right_inv := fun x ↦ by
          ext
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      ⊢ Eq φ ψ
    -/
          obtain ⟨y, -, hy⟩ := x.2
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      ⊢ Eq φ ψ
    -/
          simpa using hy ▸ NNReal.coe_nonneg y
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      h0' : Eq (↑0) 0
      ⊢ Eq φ ψ
    -/
        continuous_toFun := continuous_coe.subtype_map (by simp [s'])
        continuous_invFun := continuous_real_toNNReal.subtype_map (by simp [s']) }
    let _inst₁ : Zero s' := ⟨0, ⟨0, h0 ▸ Subtype.property (0 : s), coe_zero⟩⟩
    have h0' : ((0 : s') : ℝ) = 0 := rfl
    have e0 : e 0 = 0 := by ext; simp [e, h0, h0']
    have e0' : e.symm 0 = 0 := by
      simpa only [Homeomorph.symm_apply_apply] using congr(e.symm $(e0)).symm
    have (ξ : C(s, ℝ≥0)₀ →⋆ₙₐ[ℝ≥0] A) (hξ : Continuous ξ) :
        (let ξ' := ξ.realContinuousMapZeroOfNNReal.comp <|
          ContinuousMapZero.nonUnitalStarAlgHom_precomp ℝ ⟨e, e0⟩;
          Continuous ξ' ∧ ξ' (ContinuousMapZero.id h0') = ξ (ContinuousMapZero.id h0)) := by
      intro ξ'
    /-
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      h0' : Eq (↑0) 0
      e0 : Eq (e 0) 0
      e0' : Eq (e.symm 0) 0
      this :
        ∀ (ξ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUni …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMapZero.id h0')) (ξ (ContinuousM …
      ⊢ Eq φ ψ
    -/
      refine ⟨ξ.continuous_realContinuousMapZeroOfNNReal hξ |>.comp <| ?_, ?_⟩
    /-
      case intro
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      h0' : Eq (↑0) 0
      e0 : Eq (e 0) 0
      e0' : Eq (e.symm 0) 0
      this :
        ∀ (ξ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUni …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMapZero.id h0')) (ξ (ContinuousM …
      hφ' : Continuous ⇑(φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hφ_id : Eq ((φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      ⊢ Eq φ ψ
    -/
      · rw [continuous_induced_rng]
    /-
      case intro.intro
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      h0' : Eq (↑0) 0
      e0 : Eq (e 0) 0
      e0' : Eq (e.symm 0) 0
      this :
        ∀ (ξ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUni …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMapZero.id h0')) (ξ (ContinuousM …
      hφ' : Continuous ⇑(φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hφ_id : Eq ((φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      hψ' : Continuous ⇑(ψ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hψ_id : Eq ((ψ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      ⊢ Eq φ ψ
    -/
        exact ContinuousMap.continuous_precomp _ |>.comp continuous_induced_dom
      · exact ξ.realContinuousMapZeroOfNNReal_apply_comp_toReal (.id h0)
    obtain ⟨hφ', hφ_id⟩ := this φ hφ
    obtain ⟨hψ', hψ_id⟩ := this ψ hψ
    have hs' : CompactSpace s' := e.compactSpace
    have h' := UniqueNonUnitalContinuousFunctionalCalculus.eq_of_continuous_of_map_id
      s' h0' _ _ hφ' hψ' (hφ_id ▸ hψ_id ▸ h)
    have h'' := congr($(h').comp <|
      ContinuousMapZero.nonUnitalStarAlgHom_precomp ℝ ⟨(e.symm : C(s', s)), e0'⟩)
    /-
      case intro.intro
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      h0' : Eq (↑0) 0
      e0 : Eq (e 0) 0
      e0' : Eq (e.symm 0) 0
      this✝ :
        ∀ (ξ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUni …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMapZero.id h0')) (ξ (ContinuousM …
      hφ' : Continuous ⇑(φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hφ_id : Eq ((φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      hψ' : Continuous ⇑(ψ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hψ_id : Eq ((ψ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      hs' : CompactSpace ↑s'
      h' : Eq (φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnitalStar …
      h'' : Eq ((φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnitalSt …
      this : Eq ((ContinuousMapZero.nonUnitalStarAlgHom_precomp Real { toContinuousM …
      ⊢ Eq φ ψ
    -/
    have : (ContinuousMapZero.nonUnitalStarAlgHom_precomp ℝ ⟨(e : C(s, s')), e0⟩).comp
    /-
      case intro.intro
      X : Type u_1
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : Zero X
      A : Type u_2
      inst✝⁷ : NonUnitalRing A
      inst✝⁶ : StarRing A
      inst✝⁵ : Module Real A
      inst✝⁴ : TopologicalSpace A
      inst✝³ : TopologicalRing A
      inst✝² : IsScalarTower Real A A
      inst✝¹ : SMulCommClass Real A A
      inst✝ : UniqueNonUnitalContinuousFunctionalCalculus Real A
      s : Set NNReal
      hs : CompactSpace ↑s
      _inst : Zero ↑s
      h0 : Eq (↑0) 0
      φ ψ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A
      hφ : Continuous ⇑φ
      hψ : Continuous ⇑ψ
      h : Eq (φ { toContinuousMap := ContinuousMap.restrict s (ContinuousMap.id NNRe …
      s' : Set Real := Set.image NNReal.toReal s
      e : Homeomorph ↑s ↑s' := { toFun := Subtype.map NNReal.toReal ⋯, invFun := Sub …
      _inst₁ : Zero ↑s' := { zero := ⟨0, ⋯⟩ }
      h0' : Eq (↑0) 0
      e0 : Eq (e 0) 0
      e0' : Eq (e.symm 0) 0
      this✝ :
        ∀ (ξ : NonUnitalStarAlgHom NNReal (ContinuousMapZero (↑s) NNReal) A),
          Continuous ⇑ξ →
            let ξ' := ξ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUni …
            And (Continuous ⇑ξ') (Eq (ξ' (ContinuousMapZero.id h0')) (ξ (ContinuousM …
      hφ' : Continuous ⇑(φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hφ_id : Eq ((φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      hψ' : Continuous ⇑(ψ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.non …
      hψ_id : Eq ((ψ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnital …
      hs' : CompactSpace ↑s'
      h' : Eq (φ.realContinuousMapZeroOfNNReal.comp (ContinuousMapZero.nonUnitalStar …
      this : Eq ((ContinuousMapZero.nonUnitalStarAlgHom_precomp Real { toContinuousM …
      h'' : Eq φ.realContinuousMapZeroOfNNReal ψ.realContinuousMapZeroOfNNReal
      ⊢ Eq φ ψ
    -/
        (ContinuousMapZero.nonUnitalStarAlgHom_precomp ℝ ⟨(e.symm : C(s', s)), e0'⟩) =
    /-
      🎉 no goals
    -/
        NonUnitalStarAlgHom.id _ _ := by
      ext; simp
    simp only [NonUnitalStarAlgHom.comp_assoc, this, NonUnitalStarAlgHom.comp_id] at h''
    exact NonUnitalStarAlgHom.realContinuousMapZeroOfNNReal_injective h''


include S in
/-- Non-unital star algebra homomorphisms commute with the non-unital continuous functional
calculus. -/
lemma NonUnitalStarAlgHomClass.map_cfcₙ (φ : F) (f : R → R) (a : A)
    [CompactSpace (quasispectrum R a)] (hf : ContinuousOn f (quasispectrum R a) := by cfc_cont_tac)
    (hf₀ : f 0 = 0 := by cfc_zero_tac) (hφ : Continuous φ := by fun_prop) (ha : p a := by cfc_tac)
    (hφa : q (φ a) := by cfc_tac) : φ (cfcₙ f a) = cfcₙ f (φ a) := by
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝³⁰ : CommSemiring R
    inst✝²⁹ : Nontrivial R
    inst✝²⁸ : StarRing R
    inst✝²⁷ : MetricSpace R
    inst✝²⁶ : TopologicalSemiring R
    inst✝²⁵ : ContinuousStar R
    inst✝²⁴ : CommRing S
    inst✝²³ : Algebra R S
    inst✝²² : NonUnitalRing A
    inst✝²¹ : StarRing A
    inst✝²⁰ : TopologicalSpace A
    inst✝¹⁹ : Module R A
    inst✝¹⁸ : IsScalarTower R A A
    inst✝¹⁷ : SMulCommClass R A A
    inst✝¹⁶ : NonUnitalRing B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Module R B
    inst✝¹² : IsScalarTower R B B
    inst✝¹¹ : SMulCommClass R B B
    inst✝¹⁰ : Module S A
    inst✝⁹ : Module S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus R p
    inst✝⁵ : NonUnitalContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueNonUnitalContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : NonUnitalAlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(quasispectrum R a)
    hf : autoParam (ContinuousOn f (quasispectrum R a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ⊢ Eq (φ (cfcₙ f a)) (cfcₙ f (φ a))
  -/
  let ψ : A →⋆ₙₐ[R] B := (φ : A →⋆ₙₐ[S] B).restrictScalars R
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝³⁰ : CommSemiring R
    inst✝²⁹ : Nontrivial R
    inst✝²⁸ : StarRing R
    inst✝²⁷ : MetricSpace R
    inst✝²⁶ : TopologicalSemiring R
    inst✝²⁵ : ContinuousStar R
    inst✝²⁴ : CommRing S
    inst✝²³ : Algebra R S
    inst✝²² : NonUnitalRing A
    inst✝²¹ : StarRing A
    inst✝²⁰ : TopologicalSpace A
    inst✝¹⁹ : Module R A
    inst✝¹⁸ : IsScalarTower R A A
    inst✝¹⁷ : SMulCommClass R A A
    inst✝¹⁶ : NonUnitalRing B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Module R B
    inst✝¹² : IsScalarTower R B B
    inst✝¹¹ : SMulCommClass R B B
    inst✝¹⁰ : Module S A
    inst✝⁹ : Module S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus R p
    inst✝⁵ : NonUnitalContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueNonUnitalContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : NonUnitalAlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(quasispectrum R a)
    hf : autoParam (ContinuousOn f (quasispectrum R a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : NonUnitalStarAlgHom R A B := NonUnitalStarAlgHom.restrictScalars R ↑φ
    ⊢ Eq (φ (cfcₙ f a)) (cfcₙ f (φ a))
  -/
  have : Continuous ψ := hφ
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝³⁰ : CommSemiring R
    inst✝²⁹ : Nontrivial R
    inst✝²⁸ : StarRing R
    inst✝²⁷ : MetricSpace R
    inst✝²⁶ : TopologicalSemiring R
    inst✝²⁵ : ContinuousStar R
    inst✝²⁴ : CommRing S
    inst✝²³ : Algebra R S
    inst✝²² : NonUnitalRing A
    inst✝²¹ : StarRing A
    inst✝²⁰ : TopologicalSpace A
    inst✝¹⁹ : Module R A
    inst✝¹⁸ : IsScalarTower R A A
    inst✝¹⁷ : SMulCommClass R A A
    inst✝¹⁶ : NonUnitalRing B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Module R B
    inst✝¹² : IsScalarTower R B B
    inst✝¹¹ : SMulCommClass R B B
    inst✝¹⁰ : Module S A
    inst✝⁹ : Module S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus R p
    inst✝⁵ : NonUnitalContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueNonUnitalContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : NonUnitalAlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(quasispectrum R a)
    hf : autoParam (ContinuousOn f (quasispectrum R a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : NonUnitalStarAlgHom R A B := NonUnitalStarAlgHom.restrictScalars R ↑φ
    this : Continuous ⇑ψ
    ⊢ Eq (φ (cfcₙ f a)) (cfcₙ f (φ a))
  -/
  have h_spec := NonUnitalAlgHom.quasispectrum_apply_subset' (R := R) S φ a
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝³⁰ : CommSemiring R
    inst✝²⁹ : Nontrivial R
    inst✝²⁸ : StarRing R
    inst✝²⁷ : MetricSpace R
    inst✝²⁶ : TopologicalSemiring R
    inst✝²⁵ : ContinuousStar R
    inst✝²⁴ : CommRing S
    inst✝²³ : Algebra R S
    inst✝²² : NonUnitalRing A
    inst✝²¹ : StarRing A
    inst✝²⁰ : TopologicalSpace A
    inst✝¹⁹ : Module R A
    inst✝¹⁸ : IsScalarTower R A A
    inst✝¹⁷ : SMulCommClass R A A
    inst✝¹⁶ : NonUnitalRing B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Module R B
    inst✝¹² : IsScalarTower R B B
    inst✝¹¹ : SMulCommClass R B B
    inst✝¹⁰ : Module S A
    inst✝⁹ : Module S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : NonUnitalContinuousFunctionalCalculus R p
    inst✝⁵ : NonUnitalContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueNonUnitalContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : NonUnitalAlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(quasispectrum R a)
    hf : autoParam (ContinuousOn f (quasispectrum R a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : NonUnitalStarAlgHom R A B := NonUnitalStarAlgHom.restrictScalars R ↑φ
    this : Continuous ⇑ψ
    h_spec : HasSubset.Subset (quasispectrum R (φ a)) (quasispectrum R a)
    ⊢ Eq (φ (cfcₙ f a)) (cfcₙ f (φ a))
  -/
  have hψa : q (ψ a) := hφa
  let ι : C(quasispectrum R (ψ a), quasispectrum R a)₀ :=
    ⟨⟨Set.inclusion h_spec, continuous_id.subtype_map h_spec⟩, rfl⟩
  suffices ψ.comp (cfcₙHom ha) =
      (cfcₙHom hψa).comp (ContinuousMapZero.nonUnitalStarAlgHom_precomp R ι) by
    have hf' : ContinuousOn f (quasispectrum R (ψ a)) := hf.mono h_spec
    rw [cfcₙ_apply .., cfcₙ_apply ..]
    exact DFunLike.congr_fun this _
  refine UniqueNonUnitalContinuousFunctionalCalculus.eq_of_continuous_of_map_id _ rfl _ _
    ?_ ?_ ?apply_id
  case apply_id =>
    trans cfcₙHom hψa ⟨.restrict (quasispectrum R (ψ a)) (.id R), rfl⟩
    · simp [cfcₙHom_id]
    · congr
  all_goals
    simp [ContinuousMapZero.nonUnitalStarAlgHom_precomp]
    fun_prop


/-- Non-unital star algebra homomorphisms commute with the non-unital continuous functional
calculus.  This version is specialized to `A →⋆ₙₐ[S] B` to allow for dot notation. -/
lemma NonUnitalStarAlgHom.map_cfcₙ (φ : A →⋆ₙₐ[S] B) (f : R → R) (a : A)
    [CompactSpace (quasispectrum R a)] (hf : ContinuousOn f (quasispectrum R a) := by cfc_cont_tac)
    (hf₀ : f 0 = 0 := by cfc_zero_tac) (hφ : Continuous φ := by fun_prop) (ha : p a := by cfc_tac)
    (hφa : q (φ a) := by cfc_tac) : φ (cfcₙ f a) = cfcₙ f (φ a) :=
  /-
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²⁷ : CommSemiring R
    inst✝²⁶ : Nontrivial R
    inst✝²⁵ : StarRing R
    inst✝²⁴ : MetricSpace R
    inst✝²³ : TopologicalSemiring R
    inst✝²² : ContinuousStar R
    inst✝²¹ : CommRing S
    inst✝²⁰ : Algebra R S
    inst✝¹⁹ : NonUnitalRing A
    inst✝¹⁸ : StarRing A
    inst✝¹⁷ : TopologicalSpace A
    inst✝¹⁶ : Module R A
    inst✝¹⁵ : IsScalarTower R A A
    inst✝¹⁴ : SMulCommClass R A A
    inst✝¹³ : NonUnitalRing B
    inst✝¹² : StarRing B
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : Module R B
    inst✝⁹ : IsScalarTower R B B
    inst✝⁸ : SMulCommClass R B B
    inst✝⁷ : Module S A
    inst✝⁶ : Module S B
    inst✝⁵ : IsScalarTower R S A
    inst✝⁴ : IsScalarTower R S B
    inst✝³ : NonUnitalContinuousFunctionalCalculus R p
    inst✝² : NonUnitalContinuousFunctionalCalculus R q
    inst✝¹ : UniqueNonUnitalContinuousFunctionalCalculus R B
    φ : NonUnitalStarAlgHom S A B
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(quasispectrum R a)
    hf : autoParam (ContinuousOn f (quasispectrum R a)) _auto✝
    hf₀ : autoParam (Eq (f 0) 0) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ⊢ ContinuousOn f (quasispectrum R a)
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
  NonUnitalStarAlgHomClass.map_cfcₙ φ f a
  /-
    🎉 no goals
  -/


include S in
/-- Star algebra homomorphisms commute with the continuous functional calculus. -/
lemma StarAlgHomClass.map_cfc (φ : F) (f : R → R) (a : A)
    [CompactSpace (spectrum R a)] (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac)
    (hφ : Continuous φ := by fun_prop) (ha : p a := by cfc_tac) (hφa : q (φ a) := by cfc_tac) :
    φ (cfc f a) = cfc f (φ a) := by
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²⁵ : CommSemiring R
    inst✝²⁴ : StarRing R
    inst✝²³ : MetricSpace R
    inst✝²² : TopologicalSemiring R
    inst✝²¹ : ContinuousStar R
    inst✝²⁰ : Ring A
    inst✝¹⁹ : StarRing A
    inst✝¹⁸ : TopologicalSpace A
    inst✝¹⁷ : Algebra R A
    inst✝¹⁶ : Ring B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Algebra R B
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : ContinuousFunctionalCalculus R p
    inst✝⁵ : ContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : AlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(spectrum R a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ⊢ Eq (φ (cfc f a)) (cfc f (φ a))
  -/
  let ψ : A →⋆ₐ[R] B := (φ : A →⋆ₐ[S] B).restrictScalars R
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²⁵ : CommSemiring R
    inst✝²⁴ : StarRing R
    inst✝²³ : MetricSpace R
    inst✝²² : TopologicalSemiring R
    inst✝²¹ : ContinuousStar R
    inst✝²⁰ : Ring A
    inst✝¹⁹ : StarRing A
    inst✝¹⁸ : TopologicalSpace A
    inst✝¹⁷ : Algebra R A
    inst✝¹⁶ : Ring B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Algebra R B
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : ContinuousFunctionalCalculus R p
    inst✝⁵ : ContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : AlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(spectrum R a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : StarAlgHom R A B := StarAlgHom.restrictScalars R ↑φ
    ⊢ Eq (φ (cfc f a)) (cfc f (φ a))
  -/
  have : Continuous ψ := hφ
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²⁵ : CommSemiring R
    inst✝²⁴ : StarRing R
    inst✝²³ : MetricSpace R
    inst✝²² : TopologicalSemiring R
    inst✝²¹ : ContinuousStar R
    inst✝²⁰ : Ring A
    inst✝¹⁹ : StarRing A
    inst✝¹⁸ : TopologicalSpace A
    inst✝¹⁷ : Algebra R A
    inst✝¹⁶ : Ring B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Algebra R B
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : ContinuousFunctionalCalculus R p
    inst✝⁵ : ContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : AlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(spectrum R a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : StarAlgHom R A B := StarAlgHom.restrictScalars R ↑φ
    this : Continuous ⇑ψ
    ⊢ Eq (φ (cfc f a)) (cfc f (φ a))
  -/
  have h_spec := AlgHom.spectrum_apply_subset ψ a
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²⁵ : CommSemiring R
    inst✝²⁴ : StarRing R
    inst✝²³ : MetricSpace R
    inst✝²² : TopologicalSemiring R
    inst✝²¹ : ContinuousStar R
    inst✝²⁰ : Ring A
    inst✝¹⁹ : StarRing A
    inst✝¹⁸ : TopologicalSpace A
    inst✝¹⁷ : Algebra R A
    inst✝¹⁶ : Ring B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Algebra R B
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : ContinuousFunctionalCalculus R p
    inst✝⁵ : ContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : AlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(spectrum R a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : StarAlgHom R A B := StarAlgHom.restrictScalars R ↑φ
    this : Continuous ⇑ψ
    h_spec : HasSubset.Subset (spectrum R (ψ a)) (spectrum R a)
    ⊢ Eq (φ (cfc f a)) (cfc f (φ a))
  -/
  have hψa : q (ψ a) := hφa
  let ι : C(spectrum R (ψ a), spectrum R a) :=
    ⟨Set.inclusion h_spec, continuous_id.subtype_map h_spec⟩
  suffices ψ.comp (cfcHom ha) = (cfcHom hψa).comp (ContinuousMap.compStarAlgHom' R R ι) by
    have hf' : ContinuousOn f (spectrum R (ψ a)) := hf.mono h_spec
    rw [cfc_apply .., cfc_apply ..]
    congrm($(this) ⟨_, hf.restrict⟩)
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²⁵ : CommSemiring R
    inst✝²⁴ : StarRing R
    inst✝²³ : MetricSpace R
    inst✝²² : TopologicalSemiring R
    inst✝²¹ : ContinuousStar R
    inst✝²⁰ : Ring A
    inst✝¹⁹ : StarRing A
    inst✝¹⁸ : TopologicalSpace A
    inst✝¹⁷ : Algebra R A
    inst✝¹⁶ : Ring B
    inst✝¹⁵ : StarRing B
    inst✝¹⁴ : TopologicalSpace B
    inst✝¹³ : Algebra R B
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    inst✝¹⁰ : Algebra S A
    inst✝⁹ : Algebra S B
    inst✝⁸ : IsScalarTower R S A
    inst✝⁷ : IsScalarTower R S B
    inst✝⁶ : ContinuousFunctionalCalculus R p
    inst✝⁵ : ContinuousFunctionalCalculus R q
    inst✝⁴ : UniqueContinuousFunctionalCalculus R B
    inst✝³ : FunLike F A B
    inst✝² : AlgHomClass F S A B
    inst✝¹ : StarHomClass F A B
    φ : F
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(spectrum R a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ψ : StarAlgHom R A B := StarAlgHom.restrictScalars R ↑φ
    this : Continuous ⇑ψ
    h_spec : HasSubset.Subset (spectrum R (ψ a)) (spectrum R a)
    hψa : q (ψ a)
    ι : ContinuousMap ↑(spectrum R (ψ a)) ↑(spectrum R a) := { toFun := Set.inclus …
    ⊢ Eq (ψ.comp (cfcHom ha)) ((cfcHom hψa).comp (ContinuousMap.compStarAlgHom' R  …
  -/
  refine UniqueContinuousFunctionalCalculus.eq_of_continuous_of_map_id _ _ _ ?_ ?_ ?apply_id
  case apply_id =>
    trans cfcHom hψa (.restrict (spectrum R (ψ a)) (.id R))
    · simp [cfcHom_id]
    · congr
  all_goals
    simp [ContinuousMap.compStarAlgHom']
    fun_prop


/-- Star algebra homomorphisms commute with the continuous functional calculus.
This version is specialized to `A →⋆ₐ[S] B` to allow for dot notation. -/
lemma StarAlgHom.map_cfc (φ : A →⋆ₐ[S] B) (f : R → R) (a : A) [CompactSpace (spectrum R a)]
    (hf : ContinuousOn f (spectrum R a) := by cfc_cont_tac) (hφ : Continuous φ := by fun_prop)
    (ha : p a := by cfc_tac) (hφa : q (φ a) := by cfc_tac) :
    φ (cfc f a) = cfc f (φ a) :=
  /-
    R : Type u_2
    S : Type u_3
    A : Type u_4
    B : Type u_5
    p : A → Prop
    q : B → Prop
    inst✝²² : CommSemiring R
    inst✝²¹ : StarRing R
    inst✝²⁰ : MetricSpace R
    inst✝¹⁹ : TopologicalSemiring R
    inst✝¹⁸ : ContinuousStar R
    inst✝¹⁷ : Ring A
    inst✝¹⁶ : StarRing A
    inst✝¹⁵ : TopologicalSpace A
    inst✝¹⁴ : Algebra R A
    inst✝¹³ : Ring B
    inst✝¹² : StarRing B
    inst✝¹¹ : TopologicalSpace B
    inst✝¹⁰ : Algebra R B
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    inst✝⁷ : Algebra S A
    inst✝⁶ : Algebra S B
    inst✝⁵ : IsScalarTower R S A
    inst✝⁴ : IsScalarTower R S B
    inst✝³ : ContinuousFunctionalCalculus R p
    inst✝² : ContinuousFunctionalCalculus R q
    inst✝¹ : UniqueContinuousFunctionalCalculus R B
    φ : StarAlgHom S A B
    f : R → R
    a : A
    inst✝ : CompactSpace ↑(spectrum R a)
    hf : autoParam (ContinuousOn f (spectrum R a)) _auto✝
    hφ : autoParam (Continuous ⇑φ) _auto✝
    ha : autoParam (p a) _auto✝
    hφa : autoParam (q (φ a)) _auto✝
    ⊢ ContinuousOn f (spectrum R a)
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
  StarAlgHomClass.map_cfc φ f a
  /-
    🎉 no goals
  -/


