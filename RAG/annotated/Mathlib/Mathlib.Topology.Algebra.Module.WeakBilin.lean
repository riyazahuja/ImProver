/-- The space `E` equipped with the weak topology induced by the bilinear form `B`. -/
@[nolint unusedArguments]
def WeakBilin [CommSemiring 𝕜] [AddCommMonoid E] [Module 𝕜 E] [AddCommMonoid F] [Module 𝕜 F]
    (_ : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) := E


instance instAddCommMonoid [CommSemiring 𝕜] [a : AddCommMonoid E] [Module 𝕜 E] [AddCommMonoid F]
    [Module 𝕜 F] (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) : AddCommMonoid (WeakBilin B) := a


instance instModule [CommSemiring 𝕜] [AddCommMonoid E] [m : Module 𝕜 E] [AddCommMonoid F]
    [Module 𝕜 F] (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) : Module 𝕜 (WeakBilin B) := m


instance instAddCommGroup [CommSemiring 𝕜] [a : AddCommGroup E] [Module 𝕜 E] [AddCommMonoid F]
    [Module 𝕜 F] (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) : AddCommGroup (WeakBilin B) := a


instance (priority := 100) instModule' [CommSemiring 𝕜] [CommSemiring 𝕝] [AddCommMonoid E]
    [Module 𝕜 E] [AddCommMonoid F] [Module 𝕜 F] [m : Module 𝕝 E] (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) :
    Module 𝕝 (WeakBilin B) := m


instance instIsScalarTower [CommSemiring 𝕜] [CommSemiring 𝕝] [AddCommMonoid E] [Module 𝕜 E]
    [AddCommMonoid F] [Module 𝕜 F] [SMul 𝕝 𝕜] [Module 𝕝 E] [s : IsScalarTower 𝕝 𝕜 E]
    (B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜) : IsScalarTower 𝕝 𝕜 (WeakBilin B) := s


instance instTopologicalSpace : TopologicalSpace (WeakBilin B) :=
  TopologicalSpace.induced (fun x y => B x y) Pi.topologicalSpace


/-- The coercion `(fun x y => B x y) : E → (F → 𝕜)` is continuous. -/
theorem coeFn_continuous : Continuous fun (x : WeakBilin B) y => B x y :=
  continuous_induced_dom


theorem eval_continuous (y : F) : Continuous fun x : WeakBilin B => B x y :=
  (continuous_pi_iff.mp (coeFn_continuous B)) y


theorem continuous_of_continuous_eval [TopologicalSpace α] {g : α → WeakBilin B}
    (h : ∀ y, Continuous fun a => B (g a) y) : Continuous g :=
  continuous_induced_rng.2 (continuous_pi_iff.mpr h)


/-- The coercion `(fun x y => B x y) : E → (F → 𝕜)` is an embedding. -/
theorem isEmbedding {B : E →ₗ[𝕜] F →ₗ[𝕜] 𝕜} (hB : Function.Injective B) :
    IsEmbedding fun (x : WeakBilin B) y => B x y :=
  Function.Injective.isEmbedding_induced <| LinearMap.coe_injective.comp hB


@[deprecated (since := "2024-10-26")]
alias embedding := isEmbedding


theorem tendsto_iff_forall_eval_tendsto {l : Filter α} {f : α → WeakBilin B} {x : WeakBilin B}
    (hB : Function.Injective B) :
    Tendsto f l (𝓝 x) ↔ ∀ y, Tendsto (fun i => B (f i) y) l (𝓝 (B x y)) := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : CommSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : Module 𝕜 E
    inst✝¹ : AddCommMonoid F
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    l : Filter α
    f : α → WeakBilin B
    x : WeakBilin B
    hB : Function.Injective ⇑B
    ⊢ Iff (Filter.Tendsto f l (nhds x)) (∀ (y : F), Filter.Tendsto (fun i => (B (f …
  -/
  rw [← tendsto_pi_nhds, (isEmbedding hB).tendsto_nhds_iff]
  /-
    α : Type u_1
    𝕜 : Type u_2
    E : Type u_4
    F : Type u_5
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : CommSemiring 𝕜
    inst✝³ : AddCommMonoid E
    inst✝² : Module 𝕜 E
    inst✝¹ : AddCommMonoid F
    inst✝ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    l : Filter α
    f : α → WeakBilin B
    x : WeakBilin B
    hB : Function.Injective ⇑B
    ⊢ Iff (Filter.Tendsto (Function.comp (fun x y => (B x) y) f) l (nhds fun y =>  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Addition in `WeakBilin B` is continuous. -/
instance instContinuousAdd [ContinuousAdd 𝕜] : ContinuousAdd (WeakBilin B) := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    𝕝 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : CommSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    inst✝ : ContinuousAdd 𝕜
    ⊢ ContinuousAdd (WeakBilin B)
  -/
  refine ⟨continuous_induced_rng.2 ?_⟩
  refine
    cast (congr_arg _ ?_)
      (((coeFn_continuous B).comp continuous_fst).add ((coeFn_continuous B).comp continuous_snd))
  /-
    α : Type u_1
    𝕜 : Type u_2
    𝕝 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : CommSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    inst✝ : ContinuousAdd 𝕜
    ⊢ Eq (fun x => HAdd.hAdd (Function.comp (fun x y => (B x) y) Prod.fst x) (Func …
  -/
  ext
  /-
    case h.h
    α : Type u_1
    𝕜 : Type u_2
    𝕝 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : CommSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    inst✝ : ContinuousAdd 𝕜
    x✝¹ : Prod (WeakBilin B) (WeakBilin B)
    x✝ : F
    ⊢ Eq (HAdd.hAdd (Function.comp (fun x y => (B x) y) Prod.fst x✝¹) (Function.co …
  -/
  simp only [Function.comp_apply, Pi.add_apply, map_add, LinearMap.add_apply]
  /-
    🎉 no goals
  -/


/-- Scalar multiplication by `𝕜` on `WeakBilin B` is continuous. -/
instance instContinuousSMul [ContinuousSMul 𝕜 𝕜] : ContinuousSMul 𝕜 (WeakBilin B) := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    𝕝 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : CommSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    inst✝ : ContinuousSMul 𝕜 𝕜
    ⊢ ContinuousSMul 𝕜 (WeakBilin B)
  -/
  refine ⟨continuous_induced_rng.2 ?_⟩
  /-
    α : Type u_1
    𝕜 : Type u_2
    𝕝 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : CommSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    inst✝ : ContinuousSMul 𝕜 𝕜
    ⊢ Continuous (Function.comp (fun x y => (B x) y) fun p => HSMul.hSMul p.1 p.2)
  -/
  refine cast (congr_arg _ ?_) (continuous_fst.smul ((coeFn_continuous B).comp continuous_snd))
  /-
    α : Type u_1
    𝕜 : Type u_2
    𝕝 : Type u_3
    E : Type u_4
    F : Type u_5
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : CommSemiring 𝕜
    inst✝⁴ : AddCommMonoid E
    inst✝³ : Module 𝕜 E
    inst✝² : AddCommMonoid F
    inst✝¹ : Module 𝕜 F
    B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
    inst✝ : ContinuousSMul 𝕜 𝕜
    ⊢ Eq (fun x => HSMul.hSMul x.1 (Function.comp (fun x y => (B x) y) Prod.snd x) …
  -/
  ext
  simp only [Function.comp_apply, Pi.smul_apply, LinearMap.map_smulₛₗ, RingHom.id_apply,
    LinearMap.smul_apply]


/-- `WeakBilin B` is a `TopologicalAddGroup`, meaning that addition and negation are
continuous. -/
instance instTopologicalAddGroup [ContinuousAdd 𝕜] : TopologicalAddGroup (WeakBilin B) where
                        /-
                          α : Type u_1
                          𝕜 : Type u_2
                          𝕝 : Type u_3
                          E : Type u_4
                          F : Type u_5
                          inst✝⁶ : TopologicalSpace 𝕜
                          inst✝⁵ : CommRing 𝕜
                          inst✝⁴ : AddCommGroup E
                          inst✝³ : Module 𝕜 E
                          inst✝² : AddCommGroup F
                          inst✝¹ : Module 𝕜 F
                          B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
                          inst✝ : ContinuousAdd 𝕜
                          ⊢ ContinuousAdd (WeakBilin B)
                        -/
  toContinuousAdd := by infer_instance
                        /-
                          🎉 no goals
                        -/
  continuous_neg := by
    /-
      α : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : CommRing 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      inst✝ : ContinuousAdd 𝕜
      ⊢ Continuous fun a => Neg.neg a
    -/
    refine continuous_induced_rng.2 (continuous_pi_iff.mpr fun y => ?_)
    /-
      α : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : CommRing 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      inst✝ : ContinuousAdd 𝕜
      y : F
      ⊢ Continuous fun a => Function.comp (fun x y => (B x) y) (fun a => Neg.neg a)  …
    -/
    refine cast (congr_arg _ ?_) (eval_continuous B (-y))
    /-
      α : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : CommRing 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      inst✝ : ContinuousAdd 𝕜
      y : F
      ⊢ Eq (fun x => (B x) (Neg.neg y)) fun a => Function.comp (fun x y => (B x) y)  …
    -/
    ext x
    /-
      case h
      α : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      E : Type u_4
      F : Type u_5
      inst✝⁶ : TopologicalSpace 𝕜
      inst✝⁵ : CommRing 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : AddCommGroup F
      inst✝¹ : Module 𝕜 F
      B : LinearMap (RingHom.id 𝕜) E (LinearMap (RingHom.id 𝕜) F 𝕜)
      inst✝ : ContinuousAdd 𝕜
      y : F
      x : WeakBilin B
      ⊢ Eq ((B x) (Neg.neg y)) (Function.comp (fun x y => (B x) y) (fun a => Neg.neg …
    -/
    simp only [map_neg, Function.comp_apply, LinearMap.neg_apply]
    /-
      🎉 no goals
    -/


