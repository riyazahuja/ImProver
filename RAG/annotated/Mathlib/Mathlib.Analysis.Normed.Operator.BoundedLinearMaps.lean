/-- A function `f` satisfies `IsBoundedLinearMap 𝕜 f` if it is linear and satisfies the
inequality `‖f x‖ ≤ M * ‖x‖` for some positive constant `M`. -/
structure IsBoundedLinearMap (𝕜 : Type*) [NormedField 𝕜] {E : Type*} [SeminormedAddCommGroup E]
  [NormedSpace 𝕜 E] {F : Type*} [SeminormedAddCommGroup F] [NormedSpace 𝕜 F] (f : E → F) extends
  IsLinearMap 𝕜 f : Prop where
  bound : ∃ M, 0 < M ∧ ∀ x : E, ‖f x‖ ≤ M * ‖x‖


theorem IsLinearMap.with_bound {f : E → F} (hf : IsLinearMap 𝕜 f) (M : ℝ)
    (h : ∀ x : E, ‖f x‖ ≤ M * ‖x‖) : IsBoundedLinearMap 𝕜 f :=
  ⟨hf,
    by_cases
      (fun (this : M ≤ 0) =>
        ⟨1, zero_lt_one, fun x =>
          (h x).trans <| mul_le_mul_of_nonneg_right (this.trans zero_le_one) (norm_nonneg x)⟩)
      fun (this : ¬M ≤ 0) => ⟨M, lt_of_not_ge this, h⟩⟩


/-- A continuous linear map satisfies `IsBoundedLinearMap` -/
theorem ContinuousLinearMap.isBoundedLinearMap (f : E →L[𝕜] F) : IsBoundedLinearMap 𝕜 f :=
  { f.toLinearMap.isLinear with bound := f.bound }


/-- Construct a linear map from a function `f` satisfying `IsBoundedLinearMap 𝕜 f`. -/
def toLinearMap (f : E → F) (h : IsBoundedLinearMap 𝕜 f) : E →ₗ[𝕜] F :=
  IsLinearMap.mk' _ h.toIsLinearMap


/-- Construct a continuous linear map from `IsBoundedLinearMap`. -/
def toContinuousLinearMap {f : E → F} (hf : IsBoundedLinearMap 𝕜 f) : E →L[𝕜] F :=
  { toLinearMap f hf with
    cont :=
      let ⟨C, _, hC⟩ := hf.bound
      AddMonoidHomClass.continuous_of_bound (toLinearMap f hf) C hC }


theorem zero : IsBoundedLinearMap 𝕜 fun _ : E => (0 : F) :=
                                              /-
                                                𝕜 : Type u_1
                                                inst✝⁴ : NontriviallyNormedField 𝕜
                                                E : Type u_2
                                                inst✝³ : SeminormedAddCommGroup E
                                                inst✝² : NormedSpace 𝕜 E
                                                F : Type u_3
                                                inst✝¹ : SeminormedAddCommGroup F
                                                inst✝ : NormedSpace 𝕜 F
                                                ⊢ ∀ (x : E), LE.le (Norm.norm (0 x)) (HMul.hMul 0 (Norm.norm x))
                                              -/
  (0 : E →ₗ[𝕜] F).isLinear.with_bound 0 <| by simp [le_refl]
                                              /-
                                                🎉 no goals
                                              -/


theorem id : IsBoundedLinearMap 𝕜 fun x : E => x :=
                                           /-
                                             𝕜 : Type u_1
                                             inst✝² : NontriviallyNormedField 𝕜
                                             E : Type u_2
                                             inst✝¹ : SeminormedAddCommGroup E
                                             inst✝ : NormedSpace 𝕜 E
                                             ⊢ ∀ (x : E), LE.le (Norm.norm (LinearMap.id x)) (HMul.hMul 1 (Norm.norm x))
                                           -/
  LinearMap.id.isLinear.with_bound 1 <| by simp [le_refl]
                                           /-
                                             🎉 no goals
                                           -/


theorem fst : IsBoundedLinearMap 𝕜 fun x : E × F => x.1 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    ⊢ IsBoundedLinearMap 𝕜 fun x => x.1
  -/
  refine (LinearMap.fst 𝕜 E F).isLinear.with_bound 1 fun x => ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : Prod E F
    ⊢ LE.le (Norm.norm ((LinearMap.fst 𝕜 E F) x)) (HMul.hMul 1 (Norm.norm x))
  -/
  rw [one_mul]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : Prod E F
    ⊢ LE.le (Norm.norm ((LinearMap.fst 𝕜 E F) x)) (Norm.norm x)
  -/
  exact le_max_left _ _
  /-
    🎉 no goals
  -/


theorem snd : IsBoundedLinearMap 𝕜 fun x : E × F => x.2 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    ⊢ IsBoundedLinearMap 𝕜 fun x => x.2
  -/
  refine (LinearMap.snd 𝕜 E F).isLinear.with_bound 1 fun x => ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : Prod E F
    ⊢ LE.le (Norm.norm ((LinearMap.snd 𝕜 E F) x)) (HMul.hMul 1 (Norm.norm x))
  -/
  rw [one_mul]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : Prod E F
    ⊢ LE.le (Norm.norm ((LinearMap.snd 𝕜 E F) x)) (Norm.norm x)
  -/
  exact le_max_right _ _
  /-
    🎉 no goals
  -/


theorem smul (c : 𝕜) (hf : IsBoundedLinearMap 𝕜 f) : IsBoundedLinearMap 𝕜 (c • f) :=
  let ⟨hlf, M, _, hM⟩ := hf
  (c • hlf.mk' f).isLinear.with_bound (‖c‖ * M) fun x =>
    calc
      ‖c • f x‖ = ‖c‖ * ‖f x‖ := norm_smul c (f x)
      _ ≤ ‖c‖ * (M * ‖x‖) := mul_le_mul_of_nonneg_left (hM _) (norm_nonneg _)
      _ = ‖c‖ * M * ‖x‖ := (mul_assoc _ _ _).symm


theorem neg (hf : IsBoundedLinearMap 𝕜 f) : IsBoundedLinearMap 𝕜 fun e => -f e := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    hf : IsBoundedLinearMap 𝕜 f
    ⊢ IsBoundedLinearMap 𝕜 fun e => Neg.neg (f e)
  -/
  rw [show (fun e => -f e) = fun e => (-1 : 𝕜) • f e by funext; simp]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    hf : IsBoundedLinearMap 𝕜 f
    ⊢ IsBoundedLinearMap 𝕜 fun e => HSMul.hSMul (-1) (f e)
  -/
  exact smul (-1) hf
  /-
    🎉 no goals
  -/


theorem add (hf : IsBoundedLinearMap 𝕜 f) (hg : IsBoundedLinearMap 𝕜 g) :
    IsBoundedLinearMap 𝕜 fun e => f e + g e :=
  let ⟨hlf, Mf, _, hMf⟩ := hf
  let ⟨hlg, Mg, _, hMg⟩ := hg
  (hlf.mk' _ + hlg.mk' _).isLinear.with_bound (Mf + Mg) fun x =>
    calc
      ‖f x + g x‖ ≤ Mf * ‖x‖ + Mg * ‖x‖ := norm_add_le_of_le (hMf x) (hMg x)
                                /-
                                  𝕜 : Type u_1
                                  inst✝⁴ : NontriviallyNormedField 𝕜
                                  E : Type u_2
                                  inst✝³ : SeminormedAddCommGroup E
                                  inst✝² : NormedSpace 𝕜 E
                                  F : Type u_3
                                  inst✝¹ : SeminormedAddCommGroup F
                                  inst✝ : NormedSpace 𝕜 F
                                  f g : E → F
                                  hf : IsBoundedLinearMap 𝕜 f
                                  hg : IsBoundedLinearMap 𝕜 g
                                  hlf : IsLinearMap 𝕜 f
                                  Mf : Real
                                  left✝¹ : LT.lt 0 Mf
                                  hMf : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul Mf (Norm.norm x))
                                  hlg : IsLinearMap 𝕜 g
                                  Mg : Real
                                  left✝ : LT.lt 0 Mg
                                  hMg : ∀ (x : E), LE.le (Norm.norm (g x)) (HMul.hMul Mg (Norm.norm x))
                                  x : E
                                  ⊢ LE.le (HAdd.hAdd (HMul.hMul Mf (Norm.norm x)) (HMul.hMul Mg (Norm.norm x)))  …
                                -/
      _ ≤ (Mf + Mg) * ‖x‖ := by rw [add_mul]
                                /-
                                  🎉 no goals
                                -/


theorem sub (hf : IsBoundedLinearMap 𝕜 f) (hg : IsBoundedLinearMap 𝕜 g) :
                                                  /-
                                                    𝕜 : Type u_1
                                                    inst✝⁴ : NontriviallyNormedField 𝕜
                                                    E : Type u_2
                                                    inst✝³ : SeminormedAddCommGroup E
                                                    inst✝² : NormedSpace 𝕜 E
                                                    F : Type u_3
                                                    inst✝¹ : SeminormedAddCommGroup F
                                                    inst✝ : NormedSpace 𝕜 F
                                                    f g : E → F
                                                    hf : IsBoundedLinearMap 𝕜 f
                                                    hg : IsBoundedLinearMap 𝕜 g
                                                    ⊢ IsBoundedLinearMap 𝕜 fun e => HSub.hSub (f e) (g e)
                                                  -/
    IsBoundedLinearMap 𝕜 fun e => f e - g e := by simpa [sub_eq_add_neg] using add hf (neg hg)
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem comp {g : F → G} (hg : IsBoundedLinearMap 𝕜 g) (hf : IsBoundedLinearMap 𝕜 f) :
    IsBoundedLinearMap 𝕜 (g ∘ f) :=
  (hg.toContinuousLinearMap.comp hf.toContinuousLinearMap).isBoundedLinearMap


protected theorem tendsto (x : E) (hf : IsBoundedLinearMap 𝕜 f) : Tendsto f (𝓝 x) (𝓝 (f x)) :=
  let ⟨hf, M, _, hM⟩ := hf
  tendsto_iff_norm_sub_tendsto_zero.2 <|
    squeeze_zero (fun _ => norm_nonneg _)
      (fun e =>
        calc
                                                 /-
                                                   𝕜 : Type u_1
                                                   inst✝⁴ : NontriviallyNormedField 𝕜
                                                   E : Type u_2
                                                   inst✝³ : SeminormedAddCommGroup E
                                                   inst✝² : NormedSpace 𝕜 E
                                                   F : Type u_3
                                                   inst✝¹ : SeminormedAddCommGroup F
                                                   inst✝ : NormedSpace 𝕜 F
                                                   f : E → F
                                                   x : E
                                                   hf✝ : IsBoundedLinearMap 𝕜 f
                                                   hf : IsLinearMap 𝕜 f
                                                   M : Real
                                                   left✝ : LT.lt 0 M
                                                   hM : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul M (Norm.norm x))
                                                   e : E
                                                   ⊢ Eq (Norm.norm (HSub.hSub (f e) (f x))) (Norm.norm ((IsLinearMap.mk' f hf) (H …
                                                 -/
          ‖f e - f x‖ = ‖hf.mk' f (e - x)‖ := by rw [(hf.mk' _).map_sub e x]; rfl
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
          _ ≤ M * ‖e - x‖ := hM (e - x)
          )
                                                                        /-
                                                                          𝕜 : Type u_1
                                                                          inst✝⁴ : NontriviallyNormedField 𝕜
                                                                          E : Type u_2
                                                                          inst✝³ : SeminormedAddCommGroup E
                                                                          inst✝² : NormedSpace 𝕜 E
                                                                          F : Type u_3
                                                                          inst✝¹ : SeminormedAddCommGroup F
                                                                          inst✝ : NormedSpace 𝕜 F
                                                                          f : E → F
                                                                          x : E
                                                                          hf✝ : IsBoundedLinearMap 𝕜 f
                                                                          hf : IsLinearMap 𝕜 f
                                                                          M : Real
                                                                          left✝ : LT.lt 0 M
                                                                          hM : ∀ (x : E), LE.le (Norm.norm (f x)) (HMul.hMul M (Norm.norm x))
                                                                          this : Filter.Tendsto (fun e => HMul.hMul M (Norm.norm (HSub.hSub e x))) (nhds …
                                                                          ⊢ Filter.Tendsto (fun e => HMul.hMul M (Norm.norm (HSub.hSub e x))) (nhds x) ( …
                                                                        -/
      (suffices Tendsto (fun e : E => M * ‖e - x‖) (𝓝 x) (𝓝 (M * 0)) by simpa
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
      tendsto_const_nhds.mul (tendsto_norm_sub_self _))


theorem continuous (hf : IsBoundedLinearMap 𝕜 f) : Continuous f :=
  continuous_iff_continuousAt.2 fun _ => hf.tendsto _


theorem lim_zero_bounded_linear_map (hf : IsBoundedLinearMap 𝕜 f) : Tendsto f (𝓝 0) (𝓝 0) :=
  (hf.1.mk' _).map_zero ▸ continuous_iff_continuousAt.1 hf.continuous 0


theorem isBigO_id {f : E → F} (h : IsBoundedLinearMap 𝕜 f) (l : Filter E) : f =O[l] fun x => x :=
  let ⟨_, _, hM⟩ := h.bound
  IsBigO.of_bound _ (mem_of_superset univ_mem fun x _ => hM x)


theorem isBigO_comp {E : Type*} {g : F → G} (hg : IsBoundedLinearMap 𝕜 g) {f : E → F}
    (l : Filter E) : (fun x' => g (f x')) =O[l] f :=
  (hg.isBigO_id ⊤).comp_tendsto le_top


theorem isBigO_sub {f : E → F} (h : IsBoundedLinearMap 𝕜 f) (l : Filter E) (x : E) :
    (fun x' => f (x' - x)) =O[l] fun x' => x' - x :=
  isBigO_comp h l


/-- Taking the cartesian product of two continuous multilinear maps is a bounded linear
operation. -/
theorem isBoundedLinearMap_prod_multilinear {E : ι → Type*} [∀ i, SeminormedAddCommGroup (E i)]
    [∀ i, NormedSpace 𝕜 (E i)] :
    IsBoundedLinearMap 𝕜 fun p : ContinuousMultilinearMap 𝕜 E F × ContinuousMultilinearMap 𝕜 E G =>
      p.1.prod p.2 :=
  (ContinuousMultilinearMap.prodL 𝕜 E F G).toContinuousLinearEquiv
    |>.toContinuousLinearMap.isBoundedLinearMap


/-- Given a fixed continuous linear map `g`, associating to a continuous multilinear map `f` the
continuous multilinear map `f (g m₁, ..., g mₙ)` is a bounded linear operation. -/
theorem isBoundedLinearMap_continuousMultilinearMap_comp_linear (g : G →L[𝕜] E) :
    IsBoundedLinearMap 𝕜 fun f : ContinuousMultilinearMap 𝕜 (fun _ : ι => E) F =>
      f.compContinuousLinearMap fun _ => g :=
  (ContinuousMultilinearMap.compContinuousLinearMapL (ι := ι) (G := F) (fun _ ↦ g))
    |>.isBoundedLinearMap


theorem map_add₂ (f : M →SL[ρ₁₂] F →SL[σ₁₂] G') (x x' : M) (y : F) :
                                        /-
                                          𝕜 : Type u_1
                                          inst✝¹² : NontriviallyNormedField 𝕜
                                          F : Type u_3
                                          inst✝¹¹ : SeminormedAddCommGroup F
                                          inst✝¹⁰ : NormedSpace 𝕜 F
                                          R : Type u_5
                                          𝕜₂ : Type u_6
                                          𝕜' : Type u_7
                                          inst✝⁹ : NontriviallyNormedField 𝕜'
                                          inst✝⁸ : NontriviallyNormedField 𝕜₂
                                          M : Type u_8
                                          inst✝⁷ : TopologicalSpace M
                                          σ₁₂ : RingHom 𝕜 𝕜₂
                                          G' : Type u_9
                                          inst✝⁶ : SeminormedAddCommGroup G'
                                          inst✝⁵ : NormedSpace 𝕜₂ G'
                                          inst✝⁴ : NormedSpace 𝕜' G'
                                          inst✝³ : SMulCommClass 𝕜₂ 𝕜' G'
                                          inst✝² : Semiring R
                                          inst✝¹ : AddCommMonoid M
                                          inst✝ : Module R M
                                          ρ₁₂ : RingHom R 𝕜'
                                          f : ContinuousLinearMap ρ₁₂ M (ContinuousLinearMap σ₁₂ F G')
                                          x x' : M
                                          y : F
                                          ⊢ Eq ((f (HAdd.hAdd x x')) y) (HAdd.hAdd ((f x) y) ((f x') y))
                                        -/
    f (x + x') y = f x y + f x' y := by rw [f.map_add, add_apply]
                                        /-
                                          🎉 no goals
                                        -/


theorem map_zero₂ (f : M →SL[ρ₁₂] F →SL[σ₁₂] G') (y : F) : f 0 y = 0 := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝¹¹ : SeminormedAddCommGroup F
    inst✝¹⁰ : NormedSpace 𝕜 F
    R : Type u_5
    𝕜₂ : Type u_6
    𝕜' : Type u_7
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NontriviallyNormedField 𝕜₂
    M : Type u_8
    inst✝⁷ : TopologicalSpace M
    σ₁₂ : RingHom 𝕜 𝕜₂
    G' : Type u_9
    inst✝⁶ : SeminormedAddCommGroup G'
    inst✝⁵ : NormedSpace 𝕜₂ G'
    inst✝⁴ : NormedSpace 𝕜' G'
    inst✝³ : SMulCommClass 𝕜₂ 𝕜' G'
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    ρ₁₂ : RingHom R 𝕜'
    f : ContinuousLinearMap ρ₁₂ M (ContinuousLinearMap σ₁₂ F G')
    y : F
    ⊢ Eq ((f 0) y) 0
  -/
  rw [f.map_zero, zero_apply]
  /-
    🎉 no goals
  -/


theorem map_smulₛₗ₂ (f : M →SL[ρ₁₂] F →SL[σ₁₂] G') (c : R) (x : M) (y : F) :
                                      /-
                                        𝕜 : Type u_1
                                        inst✝¹² : NontriviallyNormedField 𝕜
                                        F : Type u_3
                                        inst✝¹¹ : SeminormedAddCommGroup F
                                        inst✝¹⁰ : NormedSpace 𝕜 F
                                        R : Type u_5
                                        𝕜₂ : Type u_6
                                        𝕜' : Type u_7
                                        inst✝⁹ : NontriviallyNormedField 𝕜'
                                        inst✝⁸ : NontriviallyNormedField 𝕜₂
                                        M : Type u_8
                                        inst✝⁷ : TopologicalSpace M
                                        σ₁₂ : RingHom 𝕜 𝕜₂
                                        G' : Type u_9
                                        inst✝⁶ : SeminormedAddCommGroup G'
                                        inst✝⁵ : NormedSpace 𝕜₂ G'
                                        inst✝⁴ : NormedSpace 𝕜' G'
                                        inst✝³ : SMulCommClass 𝕜₂ 𝕜' G'
                                        inst✝² : Semiring R
                                        inst✝¹ : AddCommMonoid M
                                        inst✝ : Module R M
                                        ρ₁₂ : RingHom R 𝕜'
                                        f : ContinuousLinearMap ρ₁₂ M (ContinuousLinearMap σ₁₂ F G')
                                        c : R
                                        x : M
                                        y : F
                                        ⊢ Eq ((f (HSMul.hSMul c x)) y) (HSMul.hSMul (ρ₁₂ c) ((f x) y))
                                      -/
    f (c • x) y = ρ₁₂ c • f x y := by rw [f.map_smulₛₗ, smul_apply]
                                      /-
                                        🎉 no goals
                                      -/


theorem map_sub₂ (f : M →SL[ρ₁₂] F →SL[σ₁₂] G') (x x' : M) (y : F) :
                                        /-
                                          𝕜 : Type u_1
                                          inst✝¹² : NontriviallyNormedField 𝕜
                                          F : Type u_3
                                          inst✝¹¹ : SeminormedAddCommGroup F
                                          inst✝¹⁰ : NormedSpace 𝕜 F
                                          R : Type u_5
                                          𝕜₂ : Type u_6
                                          𝕜' : Type u_7
                                          inst✝⁹ : NontriviallyNormedField 𝕜'
                                          inst✝⁸ : NontriviallyNormedField 𝕜₂
                                          M : Type u_8
                                          inst✝⁷ : TopologicalSpace M
                                          σ₁₂ : RingHom 𝕜 𝕜₂
                                          G' : Type u_9
                                          inst✝⁶ : SeminormedAddCommGroup G'
                                          inst✝⁵ : NormedSpace 𝕜₂ G'
                                          inst✝⁴ : NormedSpace 𝕜' G'
                                          inst✝³ : SMulCommClass 𝕜₂ 𝕜' G'
                                          inst✝² : Ring R
                                          inst✝¹ : AddCommGroup M
                                          inst✝ : Module R M
                                          ρ₁₂ : RingHom R 𝕜'
                                          f : ContinuousLinearMap ρ₁₂ M (ContinuousLinearMap σ₁₂ F G')
                                          x x' : M
                                          y : F
                                          ⊢ Eq ((f (HSub.hSub x x')) y) (HSub.hSub ((f x) y) ((f x') y))
                                        -/
    f (x - x') y = f x y - f x' y := by rw [f.map_sub, sub_apply]
                                        /-
                                          🎉 no goals
                                        -/


theorem map_neg₂ (f : M →SL[ρ₁₂] F →SL[σ₁₂] G') (x : M) (y : F) : f (-x) y = -f x y := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    F : Type u_3
    inst✝¹¹ : SeminormedAddCommGroup F
    inst✝¹⁰ : NormedSpace 𝕜 F
    R : Type u_5
    𝕜₂ : Type u_6
    𝕜' : Type u_7
    inst✝⁹ : NontriviallyNormedField 𝕜'
    inst✝⁸ : NontriviallyNormedField 𝕜₂
    M : Type u_8
    inst✝⁷ : TopologicalSpace M
    σ₁₂ : RingHom 𝕜 𝕜₂
    G' : Type u_9
    inst✝⁶ : SeminormedAddCommGroup G'
    inst✝⁵ : NormedSpace 𝕜₂ G'
    inst✝⁴ : NormedSpace 𝕜' G'
    inst✝³ : SMulCommClass 𝕜₂ 𝕜' G'
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    ρ₁₂ : RingHom R 𝕜'
    f : ContinuousLinearMap ρ₁₂ M (ContinuousLinearMap σ₁₂ F G')
    x : M
    y : F
    ⊢ Eq ((f (Neg.neg x)) y) (Neg.neg ((f x) y))
  -/
  rw [f.map_neg, neg_apply]
  /-
    🎉 no goals
  -/


theorem map_smul₂ (f : E →L[𝕜] F →L[𝕜] G) (c : 𝕜) (x : E) (y : F) : f (c • x) y = c • f x y := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
    c : 𝕜
    x : E
    y : F
    ⊢ Eq ((f (HSMul.hSMul c x)) y) (HSMul.hSMul c ((f x) y))
  -/
  rw [f.map_smul, smul_apply]
  /-
    🎉 no goals
  -/


/-- A map `f : E × F → G` satisfies `IsBoundedBilinearMap 𝕜 f` if it is bilinear and
continuous. -/
structure IsBoundedBilinearMap (f : E × F → G) : Prop where
  add_left : ∀ (x₁ x₂ : E) (y : F), f (x₁ + x₂, y) = f (x₁, y) + f (x₂, y)
  smul_left : ∀ (c : 𝕜) (x : E) (y : F), f (c • x, y) = c • f (x, y)
  add_right : ∀ (x : E) (y₁ y₂ : F), f (x, y₁ + y₂) = f (x, y₁) + f (x, y₂)
  smul_right : ∀ (c : 𝕜) (x : E) (y : F), f (x, c • y) = c • f (x, y)
  bound : ∃ C > 0, ∀ (x : E) (y : F), ‖f (x, y)‖ ≤ C * ‖x‖ * ‖y‖


theorem ContinuousLinearMap.isBoundedBilinearMap (f : E →L[𝕜] F →L[𝕜] G) :
    IsBoundedBilinearMap 𝕜 fun x : E × F => f x.1 x.2 :=
  { add_left := f.map_add₂
    smul_left := f.map_smul₂
    add_right := fun x => (f x).map_add
    smul_right := fun c x => (f x).map_smul c
    bound :=
      ⟨max ‖f‖ 1, zero_lt_one.trans_le (le_max_right _ _), fun x y =>
        (f.le_opNorm₂ x y).trans <| by
          /-
            𝕜 : Type u_1
            inst✝⁶ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁵ : SeminormedAddCommGroup E
            inst✝⁴ : NormedSpace 𝕜 E
            F : Type u_3
            inst✝³ : SeminormedAddCommGroup F
            inst✝² : NormedSpace 𝕜 F
            G : Type u_4
            inst✝¹ : SeminormedAddCommGroup G
            inst✝ : NormedSpace 𝕜 G
            f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜) F …
            x : E
            y : F
            ⊢ LE.le (HMul.hMul (HMul.hMul (Norm.norm f) (Norm.norm x)) (Norm.norm y)) (HMu …
          -/
          apply_rules [mul_le_mul_of_nonneg_right, norm_nonneg, le_max_left] ⟩ }
          /-
            🎉 no goals
          -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11445): new definition

/-- A bounded bilinear map `f : E × F → G` defines a continuous linear map
`f : E →L[𝕜] F →L[𝕜] G`. -/
def IsBoundedBilinearMap.toContinuousLinearMap (hf : IsBoundedBilinearMap 𝕜 f) :
    E →L[𝕜] F →L[𝕜] G :=
  LinearMap.mkContinuousOfExistsBound₂
    (LinearMap.mk₂ _ f.curry hf.add_left hf.smul_left hf.add_right hf.smul_right) <|
    hf.bound.imp fun _ ↦ And.right


protected theorem IsBoundedBilinearMap.isBigO (h : IsBoundedBilinearMap 𝕜 f) :
    f =O[⊤] fun p : E × F => ‖p.1‖ * ‖p.2‖ :=
  let ⟨C, _, hC⟩ := h.bound
  Asymptotics.IsBigO.of_bound C <|
                                                 /-
                                                   𝕜 : Type u_1
                                                   inst✝⁶ : NontriviallyNormedField 𝕜
                                                   E : Type u_2
                                                   inst✝⁵ : SeminormedAddCommGroup E
                                                   inst✝⁴ : NormedSpace 𝕜 E
                                                   F : Type u_3
                                                   inst✝³ : SeminormedAddCommGroup F
                                                   inst✝² : NormedSpace 𝕜 F
                                                   G : Type u_4
                                                   inst✝¹ : SeminormedAddCommGroup G
                                                   inst✝ : NormedSpace 𝕜 G
                                                   f : Prod E F → G
                                                   h : IsBoundedBilinearMap 𝕜 f
                                                   C : Real
                                                   left✝ : GT.gt C 0
                                                   hC : ∀ (x : E) (y : F), LE.le (Norm.norm (f { fst := x, snd := y })) (HMul.hMu …
                                                   x✝ : Prod E F
                                                   x : E
                                                   y : F
                                                   ⊢ LE.le (Norm.norm (f { fst := x, snd := y })) (HMul.hMul C (Norm.norm (HMul.h …
                                                 -/
    Filter.Eventually.of_forall fun ⟨x, y⟩ => by simpa [mul_assoc] using hC x y
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem IsBoundedBilinearMap.isBigO_comp {α : Type*} (H : IsBoundedBilinearMap 𝕜 f) {g : α → E}
    {h : α → F} {l : Filter α} : (fun x => f (g x, h x)) =O[l] fun x => ‖g x‖ * ‖h x‖ :=
  H.isBigO.comp_tendsto le_top


protected theorem IsBoundedBilinearMap.isBigO' (h : IsBoundedBilinearMap 𝕜 f) :
    f =O[⊤] fun p : E × F => ‖p‖ * ‖p‖ :=
  h.isBigO.trans <|
    (@Asymptotics.isBigO_fst_prod' _ E F _ _ _ _).norm_norm.mul
      (@Asymptotics.isBigO_snd_prod' _ E F _ _ _ _).norm_norm


theorem IsBoundedBilinearMap.map_sub_left (h : IsBoundedBilinearMap 𝕜 f) {x y : E} {z : F} :
    f (x - y, z) = f (x, z) - f (y, z) :=
  (h.toContinuousLinearMap.flip z).map_sub x y


theorem IsBoundedBilinearMap.map_sub_right (h : IsBoundedBilinearMap 𝕜 f) {x : E} {y z : F} :
    f (x, y - z) = f (x, y) - f (x, z) :=
  (h.toContinuousLinearMap x).map_sub y z


open Asymptotics in
/-- Useful to use together with `Continuous.comp₂`. -/
theorem IsBoundedBilinearMap.continuous (h : IsBoundedBilinearMap 𝕜 f) : Continuous f := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : Prod E F → G
    h : IsBoundedBilinearMap 𝕜 f
    ⊢ Continuous f
  -/
  refine continuous_iff_continuousAt.2 fun x ↦ tendsto_sub_nhds_zero_iff.1 ?_
  suffices Tendsto (fun y : E × F ↦ f (y.1 - x.1, y.2) + f (x.1, y.2 - x.2)) (𝓝 x) (𝓝 (0 + 0)) by
    simpa only [h.map_sub_left, h.map_sub_right, sub_add_sub_cancel, zero_add] using this
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : SeminormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : SeminormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : SeminormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : Prod E F → G
    h : IsBoundedBilinearMap 𝕜 f
    x : Prod E F
    ⊢ Filter.Tendsto (fun y => HAdd.hAdd (f { fst := HSub.hSub y.1 x.1, snd := y.2 …
  -/
  apply Tendsto.add
    /-
      case hf
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : SeminormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : Prod E F → G
      h : IsBoundedBilinearMap 𝕜 f
      x : Prod E F
      ⊢ Filter.Tendsto (fun x_1 => f { fst := HSub.hSub x_1.1 x.1, snd := x_1.2 }) ( …
    -/
  · rw [← isLittleO_one_iff ℝ, ← one_mul 1]
    /-
      case hf
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : SeminormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : Prod E F → G
      h : IsBoundedBilinearMap 𝕜 f
      x : Prod E F
      ⊢ Asymptotics.IsLittleO (nhds x) (fun x_1 => f { fst := HSub.hSub x_1.1 x.1, s …
    -/
    refine h.isBigO_comp.trans_isLittleO ?_
    /-
      case hf
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : SeminormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : Prod E F → G
      h : IsBoundedBilinearMap 𝕜 f
      x : Prod E F
      ⊢ Asymptotics.IsLittleO (nhds x) (fun x_1 => HMul.hMul (Norm.norm (HSub.hSub x …
    -/
    refine (IsLittleO.norm_left ?_).mul_isBigO (IsBigO.norm_left ?_)
      /-
        case hf.refine_1
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : SeminormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : SeminormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : SeminormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : Prod E F → G
        h : IsBoundedBilinearMap 𝕜 f
        x : Prod E F
        ⊢ Asymptotics.IsLittleO (nhds x) (fun x_1 => HSub.hSub x_1.1 x.1) fun _x => 1
      -/
    · exact (isLittleO_one_iff _).2 (tendsto_sub_nhds_zero_iff.2 (continuous_fst.tendsto _))
      /-
        🎉 no goals
      -/
      /-
        case hf.refine_2
        𝕜 : Type u_1
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝⁵ : SeminormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        F : Type u_3
        inst✝³ : SeminormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        G : Type u_4
        inst✝¹ : SeminormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        f : Prod E F → G
        h : IsBoundedBilinearMap 𝕜 f
        x : Prod E F
        ⊢ Asymptotics.IsBigO (nhds x) Prod.snd fun _x => 1
      -/
    · exact (continuous_snd.tendsto _).isBigO_one ℝ
      /-
        🎉 no goals
      -/
    /-
      case hg
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : SeminormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : Prod E F → G
      h : IsBoundedBilinearMap 𝕜 f
      x : Prod E F
      ⊢ Filter.Tendsto (fun x_1 => f { fst := x.1, snd := HSub.hSub x_1.2 x.2 }) (nh …
    -/
  · refine Continuous.tendsto' ?_ _ _ (by rw [h.map_sub_right, sub_self])
    /-
      case hg
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁵ : SeminormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝³ : SeminormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      G : Type u_4
      inst✝¹ : SeminormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : Prod E F → G
      h : IsBoundedBilinearMap 𝕜 f
      x : Prod E F
      ⊢ Continuous fun x_1 => f { fst := x.1, snd := HSub.hSub x_1.2 x.2 }
    -/
    exact ((h.toContinuousLinearMap x.1).continuous).comp (continuous_snd.sub continuous_const)
    /-
      🎉 no goals
    -/


theorem IsBoundedBilinearMap.continuous_left (h : IsBoundedBilinearMap 𝕜 f) {e₂ : F} :
    Continuous fun e₁ => f (e₁, e₂) :=
  h.continuous.comp (continuous_id.prod_mk continuous_const)


theorem IsBoundedBilinearMap.continuous_right (h : IsBoundedBilinearMap 𝕜 f) {e₁ : E} :
    Continuous fun e₂ => f (e₁, e₂) :=
  h.continuous.comp (continuous_const.prod_mk continuous_id)


/-- Useful to use together with `Continuous.comp₂`. -/
theorem ContinuousLinearMap.continuous₂ (f : E →L[𝕜] F →L[𝕜] G) :
    Continuous (Function.uncurry fun x y => f x y) :=
  f.isBoundedBilinearMap.continuous


theorem IsBoundedBilinearMap.isBoundedLinearMap_left (h : IsBoundedBilinearMap 𝕜 f) (y : F) :
    IsBoundedLinearMap 𝕜 fun x => f (x, y) :=
  (h.toContinuousLinearMap.flip y).isBoundedLinearMap


theorem IsBoundedBilinearMap.isBoundedLinearMap_right (h : IsBoundedBilinearMap 𝕜 f) (x : E) :
    IsBoundedLinearMap 𝕜 fun y => f (x, y) :=
  (h.toContinuousLinearMap x).isBoundedLinearMap


theorem isBoundedBilinearMap_smul {𝕜' : Type*} [NormedField 𝕜'] [NormedAlgebra 𝕜 𝕜'] {E : Type*}
    [SeminormedAddCommGroup E] [NormedSpace 𝕜 E] [NormedSpace 𝕜' E] [IsScalarTower 𝕜 𝕜' E] :
    IsBoundedBilinearMap 𝕜 fun p : 𝕜' × E => p.1 • p.2 :=
  (lsmul 𝕜 𝕜' : 𝕜' →L[𝕜] E →L[𝕜] E).isBoundedBilinearMap


theorem isBoundedBilinearMap_mul : IsBoundedBilinearMap 𝕜 fun p : 𝕜 × 𝕜 => p.1 * p.2 := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    ⊢ IsBoundedBilinearMap 𝕜 fun p => HMul.hMul p.1 p.2
  -/
  simp_rw [← smul_eq_mul]
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    ⊢ IsBoundedBilinearMap 𝕜 fun p => HSMul.hSMul p.1 p.2
  -/
  exact isBoundedBilinearMap_smul
  /-
    🎉 no goals
  -/


theorem isBoundedBilinearMap_comp :
    IsBoundedBilinearMap 𝕜 fun p : (F →L[𝕜] G) × (E →L[𝕜] F) => p.1.comp p.2 :=
  (compL 𝕜 E F G).isBoundedBilinearMap


theorem ContinuousLinearMap.isBoundedLinearMap_comp_left (g : F →L[𝕜] G) :
    IsBoundedLinearMap 𝕜 fun f : E →L[𝕜] F => ContinuousLinearMap.comp g f :=
  isBoundedBilinearMap_comp.isBoundedLinearMap_right _


theorem ContinuousLinearMap.isBoundedLinearMap_comp_right (f : E →L[𝕜] F) :
    IsBoundedLinearMap 𝕜 fun g : F →L[𝕜] G => ContinuousLinearMap.comp g f :=
  isBoundedBilinearMap_comp.isBoundedLinearMap_left _


theorem isBoundedBilinearMap_apply : IsBoundedBilinearMap 𝕜 fun p : (E →L[𝕜] F) × E => p.1 p.2 :=
  (ContinuousLinearMap.flip (apply 𝕜 F : E →L[𝕜] (E →L[𝕜] F) →L[𝕜] F)).isBoundedBilinearMap


/-- The function `ContinuousLinearMap.smulRight`, associating to a continuous linear map
`f : E → 𝕜` and a scalar `c : F` the tensor product `f ⊗ c` as a continuous linear map from `E` to
`F`, is a bounded bilinear map. -/
theorem isBoundedBilinearMap_smulRight :
    IsBoundedBilinearMap 𝕜 fun p =>
      (ContinuousLinearMap.smulRight : (E →L[𝕜] 𝕜) → F → E →L[𝕜] F) p.1 p.2 :=
  (smulRightL 𝕜 E F).isBoundedBilinearMap


/-- The composition of a continuous linear map with a continuous multilinear map is a bounded
bilinear operation. -/
theorem isBoundedBilinearMap_compMultilinear {ι : Type*} {E : ι → Type*} [Fintype ι]
    [∀ i, NormedAddCommGroup (E i)] [∀ i, NormedSpace 𝕜 (E i)] :
    IsBoundedBilinearMap 𝕜 fun p : (F →L[𝕜] G) × ContinuousMultilinearMap 𝕜 E F =>
      p.1.compContinuousMultilinearMap p.2 :=
  (compContinuousMultilinearMapL 𝕜 E F G).isBoundedBilinearMap


/-- Definition of the derivative of a bilinear map `f`, given at a point `p` by
`q ↦ f(p.1, q.2) + f(q.1, p.2)` as in the standard formula for the derivative of a product.
We define this function here as a linear map `E × F →ₗ[𝕜] G`, then `IsBoundedBilinearMap.deriv`
strengthens it to a continuous linear map `E × F →L[𝕜] G`.
-/
def IsBoundedBilinearMap.linearDeriv (h : IsBoundedBilinearMap 𝕜 f) (p : E × F) : E × F →ₗ[𝕜] G :=
  (h.toContinuousLinearMap.deriv₂ p).toLinearMap


/-- The derivative of a bounded bilinear map at a point `p : E × F`, as a continuous linear map
from `E × F` to `G`. The statement that this is indeed the derivative of `f` is
`IsBoundedBilinearMap.hasFDerivAt` in `Analysis.Calculus.FDeriv`. -/
def IsBoundedBilinearMap.deriv (h : IsBoundedBilinearMap 𝕜 f) (p : E × F) : E × F →L[𝕜] G :=
  h.toContinuousLinearMap.deriv₂ p


@[simp]
theorem IsBoundedBilinearMap.deriv_apply (h : IsBoundedBilinearMap 𝕜 f) (p q : E × F) :
    h.deriv p q = f (p.1, q.2) + f (q.1, p.2) :=
  rfl


/-- The function `ContinuousLinearMap.mulLeftRight : 𝕜' × 𝕜' → (𝕜' →L[𝕜] 𝕜')` is a bounded
bilinear map. -/
theorem ContinuousLinearMap.mulLeftRight_isBoundedBilinear (𝕜' : Type*) [SeminormedRing 𝕜']
    [NormedAlgebra 𝕜 𝕜'] :
    IsBoundedBilinearMap 𝕜 fun p : 𝕜' × 𝕜' => ContinuousLinearMap.mulLeftRight 𝕜 𝕜' p.1 p.2 :=
  (ContinuousLinearMap.mulLeftRight 𝕜 𝕜').isBoundedBilinearMap


/-- Given a bounded bilinear map `f`, the map associating to a point `p` the derivative of `f` at
`p` is itself a bounded linear map. -/
theorem IsBoundedBilinearMap.isBoundedLinearMap_deriv (h : IsBoundedBilinearMap 𝕜 f) :
    IsBoundedLinearMap 𝕜 fun p : E × F => h.deriv p :=
  h.toContinuousLinearMap.deriv₂.isBoundedLinearMap


@[continuity, fun_prop]
theorem Continuous.clm_comp {X} [TopologicalSpace X] {g : X → F →L[𝕜] G} {f : X → E →L[𝕜] F}
    (hg : Continuous g) (hf : Continuous f) : Continuous fun x => (g x).comp (f x) :=
  (compL 𝕜 E F G).continuous₂.comp₂ hg hf


theorem ContinuousOn.clm_comp {X} [TopologicalSpace X] {g : X → F →L[𝕜] G} {f : X → E →L[𝕜] F}
    {s : Set X} (hg : ContinuousOn g s) (hf : ContinuousOn f s) :
    ContinuousOn (fun x => (g x).comp (f x)) s :=
  (compL 𝕜 E F G).continuous₂.comp_continuousOn (hg.prod hf)


@[continuity, fun_prop]
theorem Continuous.clm_apply {X} [TopologicalSpace X] {f : X → (E →L[𝕜] F)} {g : X → E}
    (hf : Continuous f) (hg : Continuous g) : Continuous (fun x ↦ (f x) (g x)) :=
  isBoundedBilinearMap_apply.continuous.comp₂ hf hg


theorem ContinuousOn.clm_apply {X} [TopologicalSpace X] {f : X → (E →L[𝕜] F)} {g : X → E}
    {s : Set X} (hf : ContinuousOn f s) (hg : ContinuousOn g s) :
    ContinuousOn (fun x ↦ f x (g x)) s :=
  isBoundedBilinearMap_apply.continuous.comp_continuousOn (hf.prod hg)


protected theorem isOpen [CompleteSpace E] : IsOpen (range ((↑) : (E ≃L[𝕜] F) → E →L[𝕜] F)) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    ⊢ IsOpen (Set.range ContinuousLinearEquiv.toContinuousLinearMap)
  -/
  rw [isOpen_iff_mem_nhds, forall_mem_range]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    ⊢ ∀ (i : ContinuousLinearEquiv (RingHom.id 𝕜) E F), Membership.mem (nhds ↑i) ( …
  -/
  refine fun e => IsOpen.mem_nhds ?_ (mem_range_self _)
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    ⊢ IsOpen (Set.range ContinuousLinearEquiv.toContinuousLinearMap)
  -/
  let O : (E →L[𝕜] F) → E →L[𝕜] E := fun f => (e.symm : F →L[𝕜] E).comp f
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    ⊢ IsOpen (Set.range ContinuousLinearEquiv.toContinuousLinearMap)
  -/
  have h_O : Continuous O := isBoundedBilinearMap_comp.continuous_right
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    h_O : Continuous O
    ⊢ IsOpen (Set.range ContinuousLinearEquiv.toContinuousLinearMap)
  -/
  convert show IsOpen (O ⁻¹' { x | IsUnit x }) from Units.isOpen.preimage h_O using 1
  /-
    case h.e'_3
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    h_O : Continuous O
    ⊢ Eq (Set.range ContinuousLinearEquiv.toContinuousLinearMap) (Set.preimage O ( …
  -/
  ext f'
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    h_O : Continuous O
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Iff (Membership.mem (Set.range ContinuousLinearEquiv.toContinuousLinearMap)  …
  -/
  constructor
    /-
      case h.e'_3.h.mp
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
      h_O : Continuous O
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ Membership.mem (Set.range ContinuousLinearEquiv.toContinuousLinearMap) f' →  …
    -/
  · rintro ⟨e', rfl⟩
    /-
      case h.e'_3.h.mp.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
      h_O : Continuous O
      e' : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      ⊢ Membership.mem (Set.preimage O (setOf fun x => IsUnit x)) ↑e'
    -/
    exact ⟨(e'.trans e.symm).toUnit, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
      h_O : Continuous O
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      ⊢ Membership.mem (Set.preimage O (setOf fun x => IsUnit x)) f' → Membership.me …
    -/
  · rintro ⟨w, hw⟩
    /-
      case h.e'_3.h.mpr.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
      h_O : Continuous O
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      w : Units (ContinuousLinearMap (RingHom.id 𝕜) E E)
      hw : Eq (↑w) (O f')
      ⊢ Membership.mem (Set.range ContinuousLinearEquiv.toContinuousLinearMap) f'
    -/
    use (unitsEquiv 𝕜 E w).trans e
    /-
      case h
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
      h_O : Continuous O
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      w : Units (ContinuousLinearMap (RingHom.id 𝕜) E E)
      hw : Eq (↑w) (O f')
      ⊢ Eq (↑(((ContinuousLinearEquiv.unitsEquiv 𝕜 E) w).trans e)) f'
    -/
    ext x
    /-
      case h.h
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : SeminormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      O : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
      h_O : Continuous O
      f' : ContinuousLinearMap (RingHom.id 𝕜) E F
      w : Units (ContinuousLinearMap (RingHom.id 𝕜) E E)
      hw : Eq (↑w) (O f')
      x : E
      ⊢ Eq (↑(((ContinuousLinearEquiv.unitsEquiv 𝕜 E) w).trans e) x) (f' x)
    -/
    simp [O, hw]
    /-
      🎉 no goals
    -/


protected theorem nhds [CompleteSpace E] (e : E ≃L[𝕜] F) :
    range ((↑) : (E ≃L[𝕜] F) → E →L[𝕜] F) ∈ 𝓝 (e : E →L[𝕜] F) :=
                                                   /-
                                                     𝕜 : Type u_1
                                                     inst✝⁵ : NontriviallyNormedField 𝕜
                                                     E : Type u_2
                                                     inst✝⁴ : NormedAddCommGroup E
                                                     inst✝³ : NormedSpace 𝕜 E
                                                     F : Type u_3
                                                     inst✝² : SeminormedAddCommGroup F
                                                     inst✝¹ : NormedSpace 𝕜 F
                                                     inst✝ : CompleteSpace E
                                                     e : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                     ⊢ Membership.mem (Set.range ContinuousLinearEquiv.toContinuousLinearMap) ↑e
                                                   -/
  IsOpen.mem_nhds ContinuousLinearEquiv.isOpen (by simp)
                                                   /-
                                                     🎉 no goals
                                                   -/


