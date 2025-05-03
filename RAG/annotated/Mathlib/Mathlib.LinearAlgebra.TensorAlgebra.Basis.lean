/-- A basis provides an algebra isomorphism with the free algebra, replacing each basis vector
with its index. -/
noncomputable def equivFreeAlgebra (b : Basis κ R M) :
    TensorAlgebra R M ≃ₐ[R] FreeAlgebra R κ :=
  AlgEquiv.ofAlgHom
    (TensorAlgebra.lift _ (Finsupp.linearCombination _ (FreeAlgebra.ι _) ∘ₗ b.repr.toLinearMap))
    (FreeAlgebra.lift _ (ι R ∘ b))
        /-
          κ : Type uκ
          R : Type uR
          M : Type uM
          inst✝² : CommSemiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          b : Basis κ R M
          ⊢ Eq (((TensorAlgebra.lift R) ((Finsupp.linearCombination R (FreeAlgebra.ι R)) …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/
                                  /-
                                    κ : Type uκ
                                    R : Type uR
                                    M : Type uM
                                    inst✝² : CommSemiring R
                                    inst✝¹ : AddCommMonoid M
                                    inst✝ : Module R M
                                    b : Basis κ R M
                                    i : κ
                                    ⊢ Eq (((((FreeAlgebra.lift R) (Function.comp ⇑(TensorAlgebra.ι R) ⇑b)).comp (( …
                                  -/
    (hom_ext <| b.ext fun i => by simp)
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
lemma equivFreeAlgebra_ι_apply (b : Basis κ R M) (i : κ) :
    equivFreeAlgebra b (ι R (b i)) = FreeAlgebra.ι R i :=
                                               /-
                                                 κ : Type uκ
                                                 R : Type uR
                                                 M : Type uM
                                                 inst✝² : CommSemiring R
                                                 inst✝¹ : AddCommMonoid M
                                                 inst✝ : Module R M
                                                 b : Basis κ R M
                                                 i : κ
                                                 ⊢ Eq (((Finsupp.linearCombination R (FreeAlgebra.ι R)).comp ↑b.repr) (b i)) (F …
                                               -/
  (TensorAlgebra.lift_ι_apply _ _).trans <| by simp
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
lemma equivFreeAlgebra_symm_ι (b : Basis κ R M) (i : κ) :
    (equivFreeAlgebra b).symm (FreeAlgebra.ι R i) = ι R (b i) :=
  (equivFreeAlgebra b).toEquiv.symm_apply_eq.mpr <| equivFreeAlgebra_ι_apply b i |>.symm


/-- A basis on `M` can be lifted to a basis on `TensorAlgebra R M` -/
@[simps! repr_apply]
noncomputable def _root_.Basis.tensorAlgebra (b : Basis κ R M) :
    Basis (FreeMonoid κ) R (TensorAlgebra R M) :=
  (FreeAlgebra.basisFreeMonoid R κ).map <| (equivFreeAlgebra b).symm.toLinearEquiv


/-- `TensorAlgebra R M` is free when `M` is. -/
instance instModuleFree [Module.Free R M] : Module.Free R (TensorAlgebra R M) :=
  let ⟨⟨_κ, b⟩⟩ := Module.Free.exists_basis (R := R) (M := M)
  .of_basis b.tensorAlgebra


/-- The `TensorAlgebra` of a free module over a commutative semiring with no zero-divisors has
no zero-divisors. -/
instance instNoZeroDivisors [NoZeroDivisors R] [Module.Free R M] :
    NoZeroDivisors (TensorAlgebra R M) :=
  have ⟨⟨_, b⟩⟩ := ‹Module.Free R M›
  (equivFreeAlgebra b).toMulEquiv.noZeroDivisors


/-- The `TensorAlgebra` of a free module over an integral domain is a domain. -/
instance instIsDomain [IsDomain R] [Module.Free R M] : IsDomain (TensorAlgebra R M) :=
  NoZeroDivisors.to_isDomain _


open Cardinal in
lemma rank_eq [Nontrivial R] [Module.Free R M] :
    Module.rank R (TensorAlgebra R M) = Cardinal.lift.{uR} (sum fun n ↦ Module.rank R M ^ n) := by
  /-
    R : Type uR
    M : Type uM
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Nontrivial R
    inst✝ : Module.Free R M
    ⊢ Eq (Module.rank R (TensorAlgebra R M)) (Cardinal.lift.{uR, uM} (Cardinal.sum …
  -/
  let ⟨⟨κ, b⟩⟩ := Module.Free.exists_basis (R := R) (M := M)
  rw [(equivFreeAlgebra b).toLinearEquiv.rank_eq, FreeAlgebra.rank_eq, mk_list_eq_sum_pow,
    Basis.mk_eq_rank'' b]


