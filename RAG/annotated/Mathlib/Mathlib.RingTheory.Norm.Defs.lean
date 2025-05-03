/-- The norm of an element `s` of an `R`-algebra is the determinant of `(*) s`. -/
@[stacks 0BIF "Norm"]
noncomputable def norm : S →* R :=
  LinearMap.det.comp (lmul R S).toRingHom.toMonoidHom


theorem norm_apply (x : S) : norm R x = LinearMap.det (lmul R S x) := rfl


theorem norm_eq_one_of_not_exists_basis (h : ¬∃ s : Finset S, Nonempty (Basis s R S)) (x : S) :
                       /-
                         R : Type u_1
                         S : Type u_2
                         inst✝² : CommRing R
                         inst✝¹ : Ring S
                         inst✝ : Algebra R S
                         h : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
                         x : S
                         ⊢ Eq ((Algebra.norm R) x) 1
                       -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    norm R x = 1 := by rw [norm_apply, LinearMap.det]; split_ifs <;> trivial
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem norm_eq_one_of_not_module_finite (h : ¬Module.Finite R S) (x : S) : norm R x = 1 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : Not (Module.Finite R S)
    x : S
    ⊢ Eq ((Algebra.norm R) x) 1
  -/
  refine norm_eq_one_of_not_exists_basis _ (mt ?_ h) _
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : Not (Module.Finite R S)
    x : S
    ⊢ (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) R S)) …
  -/
  rintro ⟨s, ⟨b⟩⟩
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    h : Not (Module.Finite R S)
    x : S
    s : Finset S
    b : Basis (Subtype fun x => Membership.mem s x) R S
    ⊢ Module.Finite R S
  -/
  exact Module.Finite.of_basis b
  /-
    🎉 no goals
  -/

-- Can't be a `simp` lemma because it depends on a choice of basis

theorem norm_eq_matrix_det [Fintype ι] [DecidableEq ι] (b : Basis ι R S) (s : S) :
    norm R s = Matrix.det (Algebra.leftMulMatrix b s) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    ι : Type w
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    b : Basis ι R S
    s : S
    ⊢ Eq ((Algebra.norm R) s) ((Algebra.leftMulMatrix b) s).det
  -/
  rw [norm_apply, ← LinearMap.det_toMatrix b, ← toMatrix_lmul_eq]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- If `x` is in the base ring `K`, then the norm is `x ^ [L : K]`. -/
theorem norm_algebraMap_of_basis [Fintype ι] (b : Basis ι R S) (x : R) :
    norm R (algebraMap R S x) = x ^ Fintype.card ι := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    ⊢ Eq ((Algebra.norm R) ((algebraMap R S) x)) (HPow.hPow x (Fintype.card ι))
  -/
  haveI := Classical.decEq ι
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    this : DecidableEq ι
    ⊢ Eq ((Algebra.norm R) ((algebraMap R S) x)) (HPow.hPow x (Fintype.card ι))
  -/
  rw [norm_apply, ← det_toMatrix b, lmul_algebraMap]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    ι : Type w
    inst✝ : Fintype ι
    b : Basis ι R S
    x : R
    this : DecidableEq ι
    ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lsmul R R S) x)).det (HPow.hPow x (Fi …
  -/
  convert @det_diagonal _ _ _ _ _ fun _ : ι => x
    /-
      case h.e'_2.h.e'_6
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : Ring S
      inst✝¹ : Algebra R S
      ι : Type w
      inst✝ : Fintype ι
      b : Basis ι R S
      x : R
      this : DecidableEq ι
      ⊢ Eq ((LinearMap.toMatrix b b) ((Algebra.lsmul R R S) x)) (Matrix.diagonal fun …
    -/
  · ext (i j); rw [toMatrix_lsmul]
               /-
                 🎉 no goals
               -/
    /-
      case h.e'_3
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : Ring S
      inst✝¹ : Algebra R S
      ι : Type w
      inst✝ : Fintype ι
      b : Basis ι R S
      x : R
      this : DecidableEq ι
      ⊢ Eq (HPow.hPow x (Fintype.card ι)) (Finset.univ.prod fun i => x)
    -/
  · rw [Finset.prod_const, Finset.card_univ]
    /-
      🎉 no goals
    -/


/-- If `x` is in the base field `K`, then the norm is `x ^ [L : K]`.

(If `L` is not finite-dimensional over `K`, then `norm = 1 = x ^ 0 = x ^ (finrank L K)`.)
-/
@[simp]
protected theorem norm_algebraMap {L : Type*} [Ring L] [Algebra K L] (x : K) :
    norm K (algebraMap K L x) = x ^ finrank K L := by
  /-
    K : Type u_3
    inst✝² : Field K
    L : Type u_4
    inst✝¹ : Ring L
    inst✝ : Algebra K L
    x : K
    ⊢ Eq ((Algebra.norm K) ((algebraMap K L) x)) (HPow.hPow x (Module.finrank K L))
  -/
  by_cases H : ∃ s : Finset L, Nonempty (Basis s K L)
    /-
      case pos
      K : Type u_3
      inst✝² : Field K
      L : Type u_4
      inst✝¹ : Ring L
      inst✝ : Algebra K L
      x : K
      H : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) K L)
      ⊢ Eq ((Algebra.norm K) ((algebraMap K L) x)) (HPow.hPow x (Module.finrank K L))
    -/
  · rw [norm_algebraMap_of_basis H.choose_spec.some, finrank_eq_card_basis H.choose_spec.some]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_3
      inst✝² : Field K
      L : Type u_4
      inst✝¹ : Ring L
      inst✝ : Algebra K L
      x : K
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ Eq ((Algebra.norm K) ((algebraMap K L) x)) (HPow.hPow x (Module.finrank K L))
    -/
  · rw [norm_eq_one_of_not_exists_basis K H, finrank_eq_zero_of_not_exists_basis, pow_zero]
    /-
      case neg
      K : Type u_3
      inst✝² : Field K
      L : Type u_4
      inst✝¹ : Ring L
      inst✝ : Algebra K L
      x : K
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ Not (Exists fun s => Nonempty (Basis (↑↑s) K L))
    -/
    rintro ⟨s, ⟨b⟩⟩
    /-
      case neg.intro.intro
      K : Type u_3
      inst✝² : Field K
      L : Type u_4
      inst✝¹ : Ring L
      inst✝ : Algebra K L
      x : K
      H : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      s : Finset L
      b : Basis (↑↑s) K L
      ⊢ False
    -/
    exact H ⟨s, ⟨b⟩⟩
    /-
      🎉 no goals
    -/


