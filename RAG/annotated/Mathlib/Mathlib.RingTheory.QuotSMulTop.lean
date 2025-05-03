/-- An abbreviation for `M⧸rM` that keeps us from having to write
`(⊤ : Submodule R M)` over and over to satisfy the typechecker. -/
abbrev QuotSMulTop := M ⧸ r • (⊤ : Submodule R M)


/-- Reducing a module modulo `r` is the same as left tensoring with `R/(r)`. -/
noncomputable def equivQuotTensor :
    QuotSMulTop r M ≃ₗ[R] (R ⧸ Ideal.span {r}) ⊗[R] M :=
  quotEquivOfEq _ _ (ideal_span_singleton_smul _ _).symm ≪≫ₗ
   (quotTensorEquivQuotSMul M _).symm


/-- Reducing a module modulo `r` is the same as right tensoring with `R/(r)`. -/
noncomputable def equivTensorQuot :
    QuotSMulTop r M ≃ₗ[R] M ⊗[R] (R ⧸ Ideal.span {r}) :=
  quotEquivOfEq _ _ (ideal_span_singleton_smul _ _).symm ≪≫ₗ
   (tensorQuotEquivQuotSMul M _).symm


/-- The action of the functor `QuotSMulTop r` on morphisms. -/
def map : (M →ₗ[R] M') →ₗ[R] QuotSMulTop r M →ₗ[R] QuotSMulTop r M'  :=
  Submodule.mapQLinear _ _ ∘ₗ LinearMap.id.codRestrict _ fun _ =>
    map_le_iff_le_comap.mp <| le_of_eq_of_le (map_pointwise_smul _ _ _) <|
      smul_mono_right r le_top


@[simp]
lemma map_apply_mk (f : M →ₗ[R] M') (x : M) :
    map r f (Submodule.Quotient.mk x) =
      (Submodule.Quotient.mk (f x) : QuotSMulTop r M') := rfl

-- weirdly expensive to typecheck the type here?

lemma map_comp_mkQ (f : M →ₗ[R] M') :
    map r f ∘ₗ mkQ (r • ⊤) = mkQ (r • ⊤) ∘ₗ f := by
  /-
    R : Type u_2
    inst✝⁴ : CommRing R
    r : R
    M : Type u_1
    M' : Type u_3
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : AddCommGroup M'
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M M'
    ⊢ Eq (((QuotSMulTop.map r) f).comp (HSMul.hSMul r Top.top).mkQ) ((HSMul.hSMul  …
  -/
  ext; rfl
       /-
         🎉 no goals
       -/


@[simp]
lemma map_id : map r (LinearMap.id : M →ₗ[R] M) = .id :=
  DFunLike.ext _ _ <| (mkQ_surjective _).forall.mpr fun _ => rfl


@[simp]
lemma map_comp (g : M' →ₗ[R] M'') (f : M →ₗ[R] M') :
    map r (g ∘ₗ f) = map r g ∘ₗ map r f :=
  DFunLike.ext _ _ <| (mkQ_surjective _).forall.mpr fun _ => rfl


lemma equivQuotTensor_naturality_mk (f : M →ₗ[R] M') (x : M) :
    equivQuotTensor r M' (map r f (Submodule.Quotient.mk x)) =
      f.lTensor (R ⧸ Ideal.span {r})
        (equivQuotTensor r M (Submodule.Quotient.mk x)) :=
  (LinearMap.lTensor_tmul (R ⧸ Ideal.span {r}) f 1 x).symm


lemma equivQuotTensor_naturality (f : M →ₗ[R] M') :
    equivQuotTensor r M' ∘ₗ map r f =
      f.lTensor (R ⧸ Ideal.span {r}) ∘ₗ equivQuotTensor r M :=
  quot_hom_ext _ _ _ (equivQuotTensor_naturality_mk r f)


lemma equivTensorQuot_naturality_mk (f : M →ₗ[R] M') (x : M) :
    equivTensorQuot r M' (map r f (Submodule.Quotient.mk x)) =
      f.rTensor (R ⧸ Ideal.span {r})
        (equivTensorQuot r M (Submodule.Quotient.mk x)) :=
  (LinearMap.rTensor_tmul (R ⧸ Ideal.span {r}) f 1 x).symm


lemma equivTensorQuot_naturality (f : M →ₗ[R] M') :
    equivTensorQuot r M' ∘ₗ map r f =
      f.rTensor (R ⧸ Ideal.span {r}) ∘ₗ equivTensorQuot r M :=
  quot_hom_ext _ _ _ (equivTensorQuot_naturality_mk r f)


lemma map_surjective {f : M →ₗ[R] M'} (hf : Surjective f) : Surjective (map r f) :=
  have H₁ := (mkQ_surjective (r • ⊤ : Submodule R M')).comp hf
  @Surjective.of_comp _ _ _ _ (mkQ (r • ⊤ : Submodule R M)) <| by
    /-
      R : Type u_2
      inst✝⁴ : CommRing R
      r : R
      M : Type u_1
      M' : Type u_3
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : AddCommGroup M'
      inst✝ : Module R M'
      f : LinearMap (RingHom.id R) M M'
      hf : Function.Surjective ⇑f
      H₁ : Function.Surjective (Function.comp ⇑(HSMul.hSMul r Top.top).mkQ ⇑f)
      ⊢ Function.Surjective (Function.comp ⇑((QuotSMulTop.map r) f) ⇑(HSMul.hSMul r  …
    -/
    rwa [← LinearMap.coe_comp, map_comp_mkQ, LinearMap.coe_comp]
    /-
      🎉 no goals
    -/


lemma map_exact {f : M →ₗ[R] M'} {g : M' →ₗ[R] M''}
    (hfg : Exact f g) (hg : Surjective g) : Exact (map r f) (map r g) :=
  (Exact.iff_of_ladder_linearEquiv (equivQuotTensor_naturality r f).symm
                             (equivQuotTensor_naturality r g).symm).mp
    (lTensor_exact (R ⧸ Ideal.span {r}) hfg hg)


/-- Tensoring on the left and applying `QuotSMulTop · r` commute. -/
noncomputable def tensorQuotSMulTopEquivQuotSMulTop :
    M ⊗[R] QuotSMulTop r M' ≃ₗ[R] QuotSMulTop r (M ⊗[R] M') :=
  (equivTensorQuot r M').lTensor M ≪≫ₗ
    (TensorProduct.assoc R M M' (R ⧸ Ideal.span {r})).symm ≪≫ₗ
      (equivTensorQuot r (M ⊗[R] M')).symm


/-- Tensoring on the right and applying `QuotSMulTop · r` commute. -/
noncomputable def quotSMulTopTensorEquivQuotSMulTop :
    QuotSMulTop r M' ⊗[R] M ≃ₗ[R] QuotSMulTop r (M' ⊗[R] M) :=
  (equivQuotTensor r M').rTensor M ≪≫ₗ
    TensorProduct.assoc R (R ⧸ Ideal.span {r}) M' M ≪≫ₗ
      (equivQuotTensor r (M' ⊗[R] M)).symm


