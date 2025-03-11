/-- The trace of an endomorphism given a basis. -/
def traceAux : (M →ₗ[R] M) →ₗ[R] R :=
  Matrix.traceLinearMap ι R R ∘ₗ ↑(LinearMap.toMatrix b b)

-- Can't be `simp` because it would cause a loop.

theorem traceAux_def (b : Basis ι R M) (f : M →ₗ[R] M) :
    traceAux R b f = Matrix.trace (LinearMap.toMatrix b b f) :=
  rfl


theorem traceAux_eq : traceAux R b = traceAux R c :=
  LinearMap.ext fun f =>
    calc
      Matrix.trace (LinearMap.toMatrix b b f) =
          Matrix.trace (LinearMap.toMatrix b b ((LinearMap.id.comp f).comp LinearMap.id)) := by
        /-
          R : Type u
          inst✝⁶ : CommSemiring R
          M : Type v
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : Module R M
          ι : Type w
          inst✝³ : DecidableEq ι
          inst✝² : Fintype ι
          κ : Type u_1
          inst✝¹ : DecidableEq κ
          inst✝ : Fintype κ
          b : Basis ι R M
          c : Basis κ R M
          f : LinearMap (RingHom.id R) M M
          ⊢ Eq ((LinearMap.toMatrix b b) f).trace ((LinearMap.toMatrix b b) ((LinearMap. …
        -/
        rw [LinearMap.id_comp, LinearMap.comp_id]
        /-
          🎉 no goals
        -/
      _ = Matrix.trace (LinearMap.toMatrix c b LinearMap.id * LinearMap.toMatrix c c f *
          LinearMap.toMatrix b c LinearMap.id) := by
        /-
          R : Type u
          inst✝⁶ : CommSemiring R
          M : Type v
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : Module R M
          ι : Type w
          inst✝³ : DecidableEq ι
          inst✝² : Fintype ι
          κ : Type u_1
          inst✝¹ : DecidableEq κ
          inst✝ : Fintype κ
          b : Basis ι R M
          c : Basis κ R M
          f : LinearMap (RingHom.id R) M M
          ⊢ Eq ((LinearMap.toMatrix b b) ((LinearMap.id.comp f).comp LinearMap.id)).trac …
        -/
        rw [LinearMap.toMatrix_comp _ c, LinearMap.toMatrix_comp _ c]
        /-
          🎉 no goals
        -/
      _ = Matrix.trace (LinearMap.toMatrix c c f * LinearMap.toMatrix b c LinearMap.id *
          LinearMap.toMatrix c b LinearMap.id) := by
        /-
          R : Type u
          inst✝⁶ : CommSemiring R
          M : Type v
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : Module R M
          ι : Type w
          inst✝³ : DecidableEq ι
          inst✝² : Fintype ι
          κ : Type u_1
          inst✝¹ : DecidableEq κ
          inst✝ : Fintype κ
          b : Basis ι R M
          c : Basis κ R M
          f : LinearMap (RingHom.id R) M M
          ⊢ Eq (HMul.hMul (HMul.hMul ((LinearMap.toMatrix c b) LinearMap.id) ((LinearMap …
        -/
        rw [Matrix.mul_assoc, Matrix.trace_mul_comm]
        /-
          🎉 no goals
        -/
      _ = Matrix.trace (LinearMap.toMatrix c c ((f.comp LinearMap.id).comp LinearMap.id)) := by
        /-
          R : Type u
          inst✝⁶ : CommSemiring R
          M : Type v
          inst✝⁵ : AddCommMonoid M
          inst✝⁴ : Module R M
          ι : Type w
          inst✝³ : DecidableEq ι
          inst✝² : Fintype ι
          κ : Type u_1
          inst✝¹ : DecidableEq κ
          inst✝ : Fintype κ
          b : Basis ι R M
          c : Basis κ R M
          f : LinearMap (RingHom.id R) M M
          ⊢ Eq (HMul.hMul (HMul.hMul ((LinearMap.toMatrix c c) f) ((LinearMap.toMatrix b …
        -/
        rw [LinearMap.toMatrix_comp _ b, LinearMap.toMatrix_comp _ c]
        /-
          🎉 no goals
        -/
                                                        /-
                                                          R : Type u
                                                          inst✝⁶ : CommSemiring R
                                                          M : Type v
                                                          inst✝⁵ : AddCommMonoid M
                                                          inst✝⁴ : Module R M
                                                          ι : Type w
                                                          inst✝³ : DecidableEq ι
                                                          inst✝² : Fintype ι
                                                          κ : Type u_1
                                                          inst✝¹ : DecidableEq κ
                                                          inst✝ : Fintype κ
                                                          b : Basis ι R M
                                                          c : Basis κ R M
                                                          f : LinearMap (RingHom.id R) M M
                                                          ⊢ Eq ((LinearMap.toMatrix c c) ((f.comp LinearMap.id).comp LinearMap.id)).trac …
                                                        -/
      _ = Matrix.trace (LinearMap.toMatrix c c f) := by rw [LinearMap.comp_id, LinearMap.comp_id]
                                                        /-
                                                          🎉 no goals
                                                        -/


open Classical in
/-- Trace of an endomorphism independent of basis. -/
def trace : (M →ₗ[R] M) →ₗ[R] R :=
  if H : ∃ s : Finset M, Nonempty (Basis s R M) then traceAux R H.choose_spec.some else 0


open Classical in
/-- Auxiliary lemma for `trace_eq_matrix_trace`. -/
theorem trace_eq_matrix_trace_of_finset {s : Finset M} (b : Basis s R M) (f : M →ₗ[R] M) :
    trace R M f = Matrix.trace (LinearMap.toMatrix b b f) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Finset M
    b : Basis (Subtype fun x => Membership.mem s x) R M
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq ((LinearMap.trace R M) f) ((LinearMap.toMatrix b b) f).trace
  -/
  have : ∃ s : Finset M, Nonempty (Basis s R M) := ⟨s, ⟨b⟩⟩
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Finset M
    b : Basis (Subtype fun x => Membership.mem s x) R M
    f : LinearMap (RingHom.id R) M M
    this : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R …
    ⊢ Eq ((LinearMap.trace R M) f) ((LinearMap.toMatrix b b) f).trace
  -/
  rw [trace, dif_pos this, ← traceAux_def]
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Finset M
    b : Basis (Subtype fun x => Membership.mem s x) R M
    f : LinearMap (RingHom.id R) M M
    this : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R …
    ⊢ Eq ((LinearMap.traceAux R ⋯.some) f) ((LinearMap.traceAux R b) f)
  -/
  congr 1
  /-
    case e_a
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Finset M
    b : Basis (Subtype fun x => Membership.mem s x) R M
    f : LinearMap (RingHom.id R) M M
    this : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R …
    ⊢ Eq (LinearMap.traceAux R ⋯.some) (LinearMap.traceAux R b)
  -/
  apply traceAux_eq
  /-
    🎉 no goals
  -/


theorem trace_eq_matrix_trace (f : M →ₗ[R] M) :
    trace R M f = Matrix.trace (LinearMap.toMatrix b b f) := by
  classical
  rw [trace_eq_matrix_trace_of_finset R b.reindexFinsetRange, ← traceAux_def, ← traceAux_def,
    traceAux_eq R b b.reindexFinsetRange]


theorem trace_mul_comm (f g : M →ₗ[R] M) : trace R M (f * g) = trace R M (g * f) := by
  classical
  by_cases H : ∃ s : Finset M, Nonempty (Basis s R M)
  · let ⟨s, ⟨b⟩⟩ := H
    simp_rw [trace_eq_matrix_trace R b, LinearMap.toMatrix_mul]
    apply Matrix.trace_mul_comm
  · rw [trace, dif_neg H, LinearMap.zero_apply, LinearMap.zero_apply]


lemma trace_mul_cycle (f g h : M →ₗ[R] M) :
    trace R M (f * g * h) = trace R M (h * f * g) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g h : LinearMap (RingHom.id R) M M
    ⊢ Eq ((LinearMap.trace R M) (HMul.hMul (HMul.hMul f g) h)) ((LinearMap.trace R …
  -/
  rw [LinearMap.trace_mul_comm, ← mul_assoc]
  /-
    🎉 no goals
  -/


lemma trace_mul_cycle' (f g h : M →ₗ[R] M) :
    trace R M (f * (g * h)) = trace R M (h * (f * g)) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g h : LinearMap (RingHom.id R) M M
    ⊢ Eq ((LinearMap.trace R M) (HMul.hMul f (HMul.hMul g h))) ((LinearMap.trace R …
  -/
  rw [← mul_assoc, LinearMap.trace_mul_comm]
  /-
    🎉 no goals
  -/


/-- The trace of an endomorphism is invariant under conjugation -/
@[simp]
theorem trace_conj (g : M →ₗ[R] M) (f : (M →ₗ[R] M)ˣ) :
    trace R M (↑f * g * ↑f⁻¹) = trace R M g := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    g : LinearMap (RingHom.id R) M M
    f : Units (LinearMap (RingHom.id R) M M)
    ⊢ Eq ((LinearMap.trace R M) (HMul.hMul (HMul.hMul (↑f) g) ↑(Inv.inv f))) ((Lin …
  -/
  rw [trace_mul_comm]
  /-
    R : Type u
    inst✝² : CommSemiring R
    M : Type v
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    g : LinearMap (RingHom.id R) M M
    f : Units (LinearMap (RingHom.id R) M M)
    ⊢ Eq ((LinearMap.trace R M) (HMul.hMul (↑(Inv.inv f)) (HMul.hMul (↑f) g))) ((L …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma trace_lie {R M : Type*} [CommRing R] [AddCommGroup M] [Module R M] (f g : Module.End R M) :
    trace R M ⁅f, g⁆ = 0 := by
  /-
    R : Type u_2
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    ⊢ Eq ((LinearMap.trace R M) (Bracket.bracket f g)) 0
  -/
  rw [Ring.lie_def, map_sub, trace_mul_comm]
  /-
    R : Type u_2
    M : Type u_3
    inst✝² : CommRing R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f g : Module.End R M
    ⊢ Eq (HSub.hSub ((LinearMap.trace R M) (HMul.hMul g f)) ((LinearMap.trace R M) …
  -/
  exact sub_self _
  /-
    🎉 no goals
  -/


/-- The trace of a linear map correspond to the contraction pairing under the isomorphism
 `End(M) ≃ M* ⊗ M`-/
theorem trace_eq_contract_of_basis [Finite ι] (b : Basis ι R M) :
    LinearMap.trace R M ∘ₗ dualTensorHom R M M = contractLeft R M := by
  classical
    cases nonempty_fintype ι
    apply Basis.ext (Basis.tensorProduct (Basis.dualBasis b) b)
    rintro ⟨i, j⟩
    simp only [Function.comp_apply, Basis.tensorProduct_apply, Basis.coe_dualBasis, coe_comp]
    rw [trace_eq_matrix_trace R b, toMatrix_dualTensorHom]
    by_cases hij : i = j
    · rw [hij]
      simp
    rw [Matrix.StdBasisMatrix.trace_zero j i (1 : R) hij]
    simp [Finsupp.single_eq_pi_single, hij]


/-- The trace of a linear map correspond to the contraction pairing under the isomorphism
 `End(M) ≃ M* ⊗ M`-/
theorem trace_eq_contract_of_basis' [Fintype ι] [DecidableEq ι] (b : Basis ι R M) :
    LinearMap.trace R M = contractLeft R M ∘ₗ (dualTensorHomEquivOfBasis b).symm.toLinearMap := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    ι : Type u_5
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R M
    ⊢ Eq (LinearMap.trace R M) ((contractLeft R M).comp ↑(dualTensorHomEquivOfBasi …
  -/
  simp [LinearEquiv.eq_comp_toLinearMap_symm, trace_eq_contract_of_basis b]
  /-
    🎉 no goals
  -/


/-- When `M` is finite free, the trace of a linear map correspond to the contraction pairing under
the isomorphism `End(M) ≃ M* ⊗ M`-/
@[simp]
theorem trace_eq_contract : LinearMap.trace R M ∘ₗ dualTensorHom R M M = contractLeft R M :=
  trace_eq_contract_of_basis (Module.Free.chooseBasis R M)


@[simp]
theorem trace_eq_contract_apply (x : Module.Dual R M ⊗[R] M) :
    (LinearMap.trace R M) ((dualTensorHom R M M) x) = contractLeft R M x := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    x : TensorProduct R (Module.Dual R M) M
    ⊢ Eq ((LinearMap.trace R M) ((dualTensorHom R M M) x)) ((contractLeft R M) x)
  -/
  rw [← comp_apply, trace_eq_contract]
  /-
    🎉 no goals
  -/


/-- When `M` is finite free, the trace of a linear map correspond to the contraction pairing under
the isomorphism `End(M) ≃ M* ⊗ M`-/
theorem trace_eq_contract' :
    LinearMap.trace R M = contractLeft R M ∘ₗ (dualTensorHomEquiv R M M).symm.toLinearMap :=
  trace_eq_contract_of_basis' (Module.Free.chooseBasis R M)


/-- The trace of the identity endomorphism is the dimension of the free module -/
@[simp]
theorem trace_one : trace R M 1 = (finrank R M : R) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    ⊢ Eq ((LinearMap.trace R M) 1) ↑(Module.finrank R M)
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : Module.Free R M
      inst✝ : Module.Finite R M
      h✝ : Subsingleton R
      ⊢ Eq ((LinearMap.trace R M) 1) ↑(Module.finrank R M)
    -/
  · simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    h✝ : Nontrivial R
    ⊢ Eq ((LinearMap.trace R M) 1) ↑(Module.finrank R M)
  -/
  have b := Module.Free.chooseBasis R M
  /-
    case inr
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    h✝ : Nontrivial R
    b : Basis (Module.Free.ChooseBasisIndex R M) R M
    ⊢ Eq ((LinearMap.trace R M) 1) ↑(Module.finrank R M)
  -/
  rw [trace_eq_matrix_trace R b, toMatrix_one, finrank_eq_card_chooseBasisIndex]
  /-
    case inr
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    h✝ : Nontrivial R
    b : Basis (Module.Free.ChooseBasisIndex R M) R M
    ⊢ Eq (Matrix.trace 1) ↑(Fintype.card (Module.Free.ChooseBasisIndex R M))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The trace of the identity endomorphism is the dimension of the free module -/
@[simp]
                                                          /-
                                                            R : Type u_1
                                                            inst✝⁴ : CommRing R
                                                            M : Type u_2
                                                            inst✝³ : AddCommGroup M
                                                            inst✝² : Module R M
                                                            inst✝¹ : Module.Free R M
                                                            inst✝ : Module.Finite R M
                                                            ⊢ Eq ((LinearMap.trace R M) LinearMap.id) ↑(Module.finrank R M)
                                                          -/
theorem trace_id : trace R M id = (finrank R M : R) := by rw [← one_eq_id, trace_one]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem trace_transpose : trace R (Module.Dual R M) ∘ₗ Module.Dual.transpose = trace R M := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    ⊢ Eq ((LinearMap.trace R (Module.Dual R M)).comp Module.Dual.transpose) (Linea …
  -/
  let e := dualTensorHomEquiv R M M
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    e : LinearEquiv (RingHom.id R) (TensorProduct R (Module.Dual R M) M) (LinearMa …
    ⊢ Eq ((LinearMap.trace R (Module.Dual R M)).comp Module.Dual.transpose) (Linea …
  -/
  have h : Function.Surjective e.toLinearMap := e.surjective
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    e : LinearEquiv (RingHom.id R) (TensorProduct R (Module.Dual R M) M) (LinearMa …
    h : Function.Surjective ⇑↑e
    ⊢ Eq ((LinearMap.trace R (Module.Dual R M)).comp Module.Dual.transpose) (Linea …
  -/
  refine (cancel_right h).1 ?_
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    e : LinearEquiv (RingHom.id R) (TensorProduct R (Module.Dual R M) M) (LinearMa …
    h : Function.Surjective ⇑↑e
    ⊢ Eq (((LinearMap.trace R (Module.Dual R M)).comp Module.Dual.transpose).comp  …
  -/
  ext f m; simp [e]
           /-
             🎉 no goals
           -/


theorem trace_prodMap :
    trace R (M × N) ∘ₗ prodMapLinear R M N M N R =
      (coprod id id : R × R →ₗ[R] R) ∘ₗ prodMap (trace R M) (trace R N) := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    ⊢ Eq ((LinearMap.trace R (Prod M N)).comp (LinearMap.prodMapLinear R M N M N R …
  -/
  let e := (dualTensorHomEquiv R M M).prod (dualTensorHomEquiv R N N)
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    e : LinearEquiv (RingHom.id R) (Prod (TensorProduct R (Module.Dual R M) M) (Te …
    ⊢ Eq ((LinearMap.trace R (Prod M N)).comp (LinearMap.prodMapLinear R M N M N R …
  -/
  have h : Function.Surjective e.toLinearMap := e.surjective
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    e : LinearEquiv (RingHom.id R) (Prod (TensorProduct R (Module.Dual R M) M) (Te …
    h : Function.Surjective ⇑↑e
    ⊢ Eq ((LinearMap.trace R (Prod M N)).comp (LinearMap.prodMapLinear R M N M N R …
  -/
  refine (cancel_right h).1 ?_
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    e : LinearEquiv (RingHom.id R) (Prod (TensorProduct R (Module.Dual R M) M) (Te …
    h : Function.Surjective ⇑↑e
    ⊢ Eq (((LinearMap.trace R (Prod M N)).comp (LinearMap.prodMapLinear R M N M N  …
  -/
  ext
  · simp only [e, dualTensorHomEquiv, LinearEquiv.coe_prod, dualTensorHomEquivOfBasis_toLinearMap,
      AlgebraTensorModule.curry_apply, curry_apply, coe_restrictScalars, coe_comp, coe_inl,
      Function.comp_apply, prodMap_apply, map_zero, prodMapLinear_apply, dualTensorHom_prodMap_zero,
      trace_eq_contract_apply, contractLeft_apply, fst_apply, coprod_apply, id_coe, id_eq, add_zero]
  · simp only [e, dualTensorHomEquiv, LinearEquiv.coe_prod, dualTensorHomEquivOfBasis_toLinearMap,
      AlgebraTensorModule.curry_apply, curry_apply, coe_restrictScalars, coe_comp, coe_inr,
      Function.comp_apply, prodMap_apply, map_zero, prodMapLinear_apply, zero_prodMap_dualTensorHom,
      trace_eq_contract_apply, contractLeft_apply, snd_apply, coprod_apply, id_coe, id_eq, zero_add]


theorem trace_prodMap' (f : M →ₗ[R] M) (g : N →ₗ[R] N) :
    trace R (M × N) (prodMap f g) = trace R M f + trace R N g := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M M
    g : LinearMap (RingHom.id R) N N
    ⊢ Eq ((LinearMap.trace R (Prod M N)) (f.prodMap g)) (HAdd.hAdd ((LinearMap.tra …
  -/
  have h := LinearMap.ext_iff.1 (trace_prodMap R M N) (f, g)
  simp only [coe_comp, Function.comp_apply, prodMap_apply, coprod_apply, id_coe, id,
    prodMapLinear_apply] at h
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M M
    g : LinearMap (RingHom.id R) N N
    h : Eq ((LinearMap.trace R (Prod M N)) (f.prodMap g)) (HAdd.hAdd ({ toFun := _ …
    ⊢ Eq ((LinearMap.trace R (Prod M N)) (f.prodMap g)) (HAdd.hAdd ((LinearMap.tra …
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem trace_tensorProduct : compr₂ (mapBilinear R M N M N) (trace R (M ⊗ N)) =
    compl₁₂ (lsmul R R : R →ₗ[R] R →ₗ[R] R) (trace R M) (trace R N) := by
  apply
    (compl₁₂_inj (show Surjective (dualTensorHom R M M) from (dualTensorHomEquiv R M M).surjective)
        (show Surjective (dualTensorHom R N N) from (dualTensorHomEquiv R N N).surjective)).1
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    ⊢ Eq (((TensorProduct.mapBilinear R M N M N).compr₂ (LinearMap.trace R (Tensor …
  -/
  ext f m g n
  simp only [AlgebraTensorModule.curry_apply, toFun_eq_coe, TensorProduct.curry_apply,
    coe_restrictScalars, compl₁₂_apply, compr₂_apply, mapBilinear_apply,
    trace_eq_contract_apply, contractLeft_apply, lsmul_apply, Algebra.id.smul_eq_mul,
    map_dualTensorHom, dualDistrib_apply]


theorem trace_comp_comm :
    compr₂ (llcomp R M N M) (trace R M) = compr₂ (llcomp R N M N).flip (trace R N) := by
  apply
    (compl₁₂_inj (show Surjective (dualTensorHom R N M) from (dualTensorHomEquiv R N M).surjective)
        (show Surjective (dualTensorHom R M N) from (dualTensorHomEquiv R M N).surjective)).1
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    ⊢ Eq (((LinearMap.llcomp R M N M).compr₂ (LinearMap.trace R M)).compl₁₂ (dualT …
  -/
  ext g m f n
  simp only [AlgebraTensorModule.curry_apply, TensorProduct.curry_apply,
    coe_restrictScalars, compl₁₂_apply, compr₂_apply, flip_apply, llcomp_apply',
    comp_dualTensorHom, LinearMapClass.map_smul, trace_eq_contract_apply,
    contractLeft_apply, smul_eq_mul, mul_comm]


@[simp]
theorem trace_transpose' (f : M →ₗ[R] M) :
    trace R _ (Module.Dual.transpose (R := R) f) = trace R M f := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : Module.Free R M
    inst✝ : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    ⊢ Eq ((LinearMap.trace R (Module.Dual R M)) (Module.Dual.transpose f)) ((Linea …
  -/
  rw [← comp_apply, trace_transpose]
  /-
    🎉 no goals
  -/


theorem trace_tensorProduct' (f : M →ₗ[R] M) (g : N →ₗ[R] N) :
    trace R (M ⊗ N) (map f g) = trace R M f * trace R N g := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M M
    g : LinearMap (RingHom.id R) N N
    ⊢ Eq ((LinearMap.trace R (TensorProduct R M N)) (TensorProduct.map f g)) (HMul …
  -/
  have h := LinearMap.ext_iff.1 (LinearMap.ext_iff.1 (trace_tensorProduct R M N) f) g
  simp only [compr₂_apply, mapBilinear_apply, compl₁₂_apply, lsmul_apply,
    Algebra.id.smul_eq_mul] at h
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M M
    g : LinearMap (RingHom.id R) N N
    h : Eq ((LinearMap.trace R (TensorProduct R M N)) (TensorProduct.map f g)) (HM …
    ⊢ Eq ((LinearMap.trace R (TensorProduct R M N)) (TensorProduct.map f g)) (HMul …
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem trace_comp_comm' (f : M →ₗ[R] N) (g : N →ₗ[R] M) :
    trace R M (g ∘ₗ f) = trace R N (f ∘ₗ g) := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N M
    ⊢ Eq ((LinearMap.trace R M) (g.comp f)) ((LinearMap.trace R N) (f.comp g))
  -/
  have h := LinearMap.ext_iff.1 (LinearMap.ext_iff.1 (trace_comp_comm R M N) g) f
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N M
    h : Eq ((((LinearMap.llcomp R M N M).compr₂ (LinearMap.trace R M)) g) f) ((((L …
    ⊢ Eq ((LinearMap.trace R M) (g.comp f)) ((LinearMap.trace R N) (f.comp g))
  -/
  simp only [llcomp_apply', compr₂_apply, flip_apply] at h
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u_3
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R N
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N M
    h : Eq ((LinearMap.trace R M) (g.comp f)) ((LinearMap.trace R N) (f.comp g))
    ⊢ Eq ((LinearMap.trace R M) (g.comp f)) ((LinearMap.trace R N) (f.comp g))
  -/
  exact h
  /-
    🎉 no goals
  -/


variable [Module.Free R N] [Module.Finite R N] [Module.Free R P] [Module.Finite R P] in
lemma trace_comp_cycle (f : M →ₗ[R] N) (g : N →ₗ[R] P) (h : P →ₗ[R] M) :
    trace R P (g ∘ₗ f ∘ₗ h) = trace R N (f ∘ₗ h ∘ₗ g) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    M : Type u_2
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R P
    inst✝³ : Module.Free R N
    inst✝² : Module.Finite R N
    inst✝¹ : Module.Free R P
    inst✝ : Module.Finite R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LinearMap (RingHom.id R) P M
    ⊢ Eq ((LinearMap.trace R P) (g.comp (f.comp h))) ((LinearMap.trace R N) (f.com …
  -/
  rw [trace_comp_comm', comp_assoc]
  /-
    🎉 no goals
  -/


variable [Module.Free R M] [Module.Finite R M] [Module.Free R P] [Module.Finite R P] in
lemma trace_comp_cycle' (f : M →ₗ[R] N) (g : N →ₗ[R] P) (h : P →ₗ[R] M) :
    trace R P ((g ∘ₗ f) ∘ₗ h) = trace R M ((h ∘ₗ g) ∘ₗ f) := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommRing R
    M : Type u_2
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : Module R M
    N : Type u_3
    P : Type u_4
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R N
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R P
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Free R P
    inst✝ : Module.Finite R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    h : LinearMap (RingHom.id R) P M
    ⊢ Eq ((LinearMap.trace R P) ((g.comp f).comp h)) ((LinearMap.trace R M) ((h.co …
  -/
  rw [trace_comp_comm', ← comp_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem trace_conj' (f : M →ₗ[R] M) (e : M ≃ₗ[R] N) : trace R N (e.conj f) = trace R M f := by
  classical
  by_cases hM : ∃ s : Finset M, Nonempty (Basis s R M)
  · obtain ⟨s, ⟨b⟩⟩ := hM
    haveI := Module.Finite.of_basis b
    haveI := (Module.free_def R M).mpr ⟨_, ⟨b⟩⟩
    haveI := Module.Finite.of_basis (b.map e)
    haveI := (Module.free_def R N).mpr ⟨_, ⟨(b.map e).reindex (e.toEquiv.image _)⟩⟩
    rw [e.conj_apply, trace_comp_comm', ← comp_assoc, LinearEquiv.comp_coe,
      LinearEquiv.self_trans_symm, LinearEquiv.refl_toLinearMap, id_comp]
  · rw [trace, trace, dif_neg hM, dif_neg ?_, zero_apply, zero_apply]
    rintro ⟨s, ⟨b⟩⟩
    exact hM ⟨s.image e.symm, ⟨(b.map e.symm).reindex
      ((e.symm.toEquiv.image s).trans (Equiv.Set.ofEq Finset.coe_image.symm))⟩⟩


theorem IsProj.trace {p : Submodule R M} {f : M →ₗ[R] M} (h : IsProj p f) [Module.Free R p]
    [Module.Finite R p] [Module.Free R (ker f)] [Module.Finite R (ker f)] :
    trace R M f = (finrank R p : R) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    p : Submodule R M
    f : LinearMap (RingHom.id R) M M
    h : LinearMap.IsProj p f
    inst✝³ : Module.Free R (Subtype fun x => Membership.mem p x)
    inst✝² : Module.Finite R (Subtype fun x => Membership.mem p x)
    inst✝¹ : Module.Free R (Subtype fun x => Membership.mem (LinearMap.ker f) x)
    inst✝ : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker f) x)
    ⊢ Eq ((LinearMap.trace R M) f) ↑(Module.finrank R (Subtype fun x => Membership …
  -/
  rw [h.eq_conj_prodMap, trace_conj', trace_prodMap', trace_id, map_zero, add_zero]
  /-
    🎉 no goals
  -/


lemma isNilpotent_trace_of_isNilpotent {f : M →ₗ[R] M} (hf : IsNilpotent f) :
    IsNilpotent (trace R M f) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    hf : IsNilpotent f
    ⊢ IsNilpotent ((LinearMap.trace R M) f)
  -/
  by_cases H : ∃ s : Finset M, Nonempty (Basis s R M)
  /-
    case pos
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    hf : IsNilpotent f
    H : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R M)
    ⊢ IsNilpotent ((LinearMap.trace R M) f)
  -/
  swap
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : LinearMap (RingHom.id R) M M
      hf : IsNilpotent f
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ IsNilpotent ((LinearMap.trace R M) f)
    -/
  · rw [LinearMap.trace, dif_neg H]
    /-
      case neg
      R : Type u_1
      inst✝² : CommRing R
      M : Type u_2
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      f : LinearMap (RingHom.id R) M M
      hf : IsNilpotent f
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ IsNilpotent (0 f)
    -/
    exact IsNilpotent.zero
    /-
      🎉 no goals
    -/
  /-
    case pos
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M M
    hf : IsNilpotent f
    H : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R M)
    ⊢ IsNilpotent ((LinearMap.trace R M) f)
  -/
  obtain ⟨s, ⟨b⟩⟩ := H
  classical
  rw [trace_eq_matrix_trace R b]
  apply Matrix.isNilpotent_trace_of_isNilpotent
  simpa


lemma trace_comp_eq_mul_of_commute_of_isNilpotent [IsReduced R] {f g : Module.End R M}
    (μ : R) (h_comm : Commute f g) (hg : IsNilpotent (g - algebraMap R _ μ)) :
    trace R M (f ∘ₗ g) = μ * trace R M f := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsReduced R
    f g : Module.End R M
    μ : R
    h_comm : Commute f g
    hg : IsNilpotent (HSub.hSub g ((algebraMap R (Module.End R M)) μ))
    ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp f g)) (HMul.hMul μ ((LinearMap.tra …
  -/
  set n := g - algebraMap R _ μ
  replace hg : trace R M (f ∘ₗ n) = 0 := by
    rw [← isNilpotent_iff_eq_zero, ← mul_eq_comp]
    refine isNilpotent_trace_of_isNilpotent (Commute.isNilpotent_mul_right ?_ hg)
    exact h_comm.sub_right (Algebra.commute_algebraMap_right μ f)
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsReduced R
    f g : Module.End R M
    μ : R
    h_comm : Commute f g
    n : Module.End R M := HSub.hSub g ((algebraMap R (Module.End R M)) μ)
    hg : Eq ((LinearMap.trace R M) (LinearMap.comp f n)) 0
    ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp f g)) (HMul.hMul μ ((LinearMap.tra …
  -/
  have hμ : g = algebraMap R _ μ + n := eq_add_of_sub_eq' rfl
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsReduced R
    f g : Module.End R M
    μ : R
    h_comm : Commute f g
    n : Module.End R M := HSub.hSub g ((algebraMap R (Module.End R M)) μ)
    hg : Eq ((LinearMap.trace R M) (LinearMap.comp f n)) 0
    hμ : Eq g (HAdd.hAdd ((algebraMap R (Module.End R M)) μ) n)
    ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp f g)) (HMul.hMul μ ((LinearMap.tra …
  -/
  have : f ∘ₗ algebraMap R _ μ = μ • f := by ext; simp -- TODO Surely exists?
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsReduced R
    f g : Module.End R M
    μ : R
    h_comm : Commute f g
    n : Module.End R M := HSub.hSub g ((algebraMap R (Module.End R M)) μ)
    hg : Eq ((LinearMap.trace R M) (LinearMap.comp f n)) 0
    hμ : Eq g (HAdd.hAdd ((algebraMap R (Module.End R M)) μ) n)
    this : Eq (LinearMap.comp f ((algebraMap R (LinearMap (RingHom.id R) M M)) μ)) …
    ⊢ Eq ((LinearMap.trace R M) (LinearMap.comp f g)) (HMul.hMul μ ((LinearMap.tra …
  -/
  rw [hμ, comp_add, map_add, hg, add_zero, this, LinearMap.map_smul, smul_eq_mul]
  /-
    🎉 no goals
  -/

-- This result requires `Mathlib.RingTheory.TensorProduct.Free`. Maybe it should move elsewhere?

@[simp]
lemma trace_baseChange [Module.Free R M] [Module.Finite R M]
    (f : M →ₗ[R] M) (A : Type*) [CommRing A] [Algebra R A] :
    trace A _ (f.baseChange A) = algebraMap R A (trace R _ f) := by
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_6
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    ⊢ Eq ((LinearMap.trace A (TensorProduct R A M)) (LinearMap.baseChange A f)) (( …
  -/
  let b := Module.Free.chooseBasis R M
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_6
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    b : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    ⊢ Eq ((LinearMap.trace A (TensorProduct R A M)) (LinearMap.baseChange A f)) (( …
  -/
  let b' := Algebra.TensorProduct.basis A b
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_6
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    b : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    b' : Basis (Module.Free.ChooseBasisIndex R M) A (TensorProduct R A M) := Algeb …
    ⊢ Eq ((LinearMap.trace A (TensorProduct R A M)) (LinearMap.baseChange A f)) (( …
  -/
  change _ = (algebraMap R A : R →+ A) _
  /-
    R : Type u_1
    inst✝⁶ : CommRing R
    M : Type u_2
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    inst✝³ : Module.Free R M
    inst✝² : Module.Finite R M
    f : LinearMap (RingHom.id R) M M
    A : Type u_6
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    b : Basis (Module.Free.ChooseBasisIndex R M) R M := Module.Free.chooseBasis R M
    b' : Basis (Module.Free.ChooseBasisIndex R M) A (TensorProduct R A M) := Algeb …
    ⊢ Eq ((LinearMap.trace A (TensorProduct R A M)) (LinearMap.baseChange A f)) (↑ …
  -/
  simp [b', trace_eq_matrix_trace R b, trace_eq_matrix_trace A b', AddMonoidHom.map_trace]
  /-
    🎉 no goals
  -/


