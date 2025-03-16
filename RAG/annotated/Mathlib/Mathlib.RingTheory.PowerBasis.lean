/-- `pb : PowerBasis R S` states that `1, pb.gen, ..., pb.gen ^ (pb.dim - 1)`
is a basis for the `R`-algebra `S` (viewed as `R`-module).

This is a structure, not a class, since the same algebra can have many power bases.
For the common case where `S` is defined by adjoining an integral element to `R`,
the canonical power basis is given by `{Algebra,IntermediateField}.adjoin.powerBasis`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure PowerBasis (R S : Type*) [CommRing R] [Ring S] [Algebra R S] where
  gen : S
  dim : ℕ
  basis : Basis (Fin dim) R S
  basis_eq_pow : ∀ (i), basis i = gen ^ (i : ℕ)

-- this is usually not needed because of `basis_eq_pow` but can be needed in some cases;
-- in such circumstances, add it manually using `@[simps dim gen basis]`.

@[simp]
theorem coe_basis (pb : PowerBasis R S) : ⇑pb.basis = fun i : Fin pb.dim => pb.gen ^ (i : ℕ) :=
  funext pb.basis_eq_pow


/-- Cannot be an instance because `PowerBasis` cannot be a class. -/
theorem finite (pb : PowerBasis R S) : Module.Finite R S := .of_basis pb.basis


@[deprecated (since := "2024-03-05")] alias finiteDimensional := PowerBasis.finite


theorem finrank [StrongRankCondition R] (pb : PowerBasis R S) :
    Module.finrank R S = pb.dim := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : StrongRankCondition R
    pb : PowerBasis R S
    ⊢ Eq (Module.finrank R S) pb.dim
  -/
  rw [Module.finrank_eq_card_basis pb.basis, Fintype.card_fin]
  /-
    🎉 no goals
  -/


theorem mem_span_pow' {x y : S} {d : ℕ} :
    y ∈ Submodule.span R (Set.range fun i : Fin d => x ^ (i : ℕ)) ↔
      ∃ f : R[X], f.degree < d ∧ y = aeval x f := by
  have : (Set.range fun i : Fin d => x ^ (i : ℕ)) = (fun i : ℕ => x ^ i) '' ↑(Finset.range d) := by
    ext n
    simp_rw [Set.mem_range, Set.mem_image, Finset.mem_coe, Finset.mem_range]
    exact ⟨fun ⟨⟨i, hi⟩, hy⟩ => ⟨i, hi, hy⟩, fun ⟨i, hi, hy⟩ => ⟨⟨i, hi⟩, hy⟩⟩
  simp only [this, mem_span_image_iff_linearCombination, degree_lt_iff_coeff_zero, Finsupp.support,
    exists_iff_exists_finsupp, coeff, aeval_def, eval₂RingHom', eval₂_eq_sum, Polynomial.sum,
    mem_supported', linearCombination, Finsupp.sum, Algebra.smul_def, eval₂_zero, exists_prop,
    LinearMap.id_coe, eval₂_one, id, not_lt, Finsupp.coe_lsum, LinearMap.coe_smulRight,
    Finset.mem_range, AlgHom.coe_mks, Finset.mem_coe]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    x y : S
    d : Nat
    this : Eq (Set.range fun i => HPow.hPow x ↑i) (Set.image (fun i => HPow.hPow x …
    ⊢ Iff (Exists fun l => And (∀ (x : Nat), LE.le d x → Eq (l x) 0) (Eq (l.suppor …
  -/
  simp_rw [@eq_comm _ y]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    x y : S
    d : Nat
    this : Eq (Set.range fun i => HPow.hPow x ↑i) (Set.image (fun i => HPow.hPow x …
    ⊢ Iff (Exists fun l => And (∀ (x : Nat), LE.le d x → Eq (l x) 0) (Eq (l.suppor …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


theorem mem_span_pow {x y : S} {d : ℕ} (hd : d ≠ 0) :
    y ∈ Submodule.span R (Set.range fun i : Fin d => x ^ (i : ℕ)) ↔
      ∃ f : R[X], f.natDegree < d ∧ y = aeval x f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    x y : S
    d : Nat
    hd : Ne d 0
    ⊢ Iff (Membership.mem (Submodule.span R (Set.range fun i => HPow.hPow x ↑i)) y …
  -/
  rw [mem_span_pow']
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    x y : S
    d : Nat
    hd : Ne d 0
    ⊢ Iff (Exists fun f => And (LT.lt f.degree ↑d) (Eq y ((Polynomial.aeval x) f)) …
  -/
  constructor <;>
      /-
        case mp
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        x y : S
        d : Nat
        hd : Ne d 0
        ⊢ (Exists fun f => And (LT.lt f.degree ↑d) (Eq y ((Polynomial.aeval x) f))) →  …
      -/
      /-
        case mp.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        x y : S
        d : Nat
        hd : Ne d 0
        f : Polynomial R
        h : LT.lt f.degree ↑d
        hy : Eq y ((Polynomial.aeval x) f)
        ⊢ Exists fun f => And (LT.lt f.natDegree d) (Eq y ((Polynomial.aeval x) f))
      -/
      /-
        case mp.intro.intro
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        x y : S
        d : Nat
        hd : Ne d 0
        f : Polynomial R
        h : LT.lt f.degree ↑d
        hy : Eq y ((Polynomial.aeval x) f)
        ⊢ LT.lt f.natDegree d
      -/
        /-
          case pos
          R : Type u_1
          S : Type u_2
          inst✝² : CommRing R
          inst✝¹ : Ring S
          inst✝ : Algebra R S
          x y : S
          d : Nat
          hd : Ne d 0
          f : Polynomial R
          h : LT.lt f.degree ↑d
          hy : Eq y ((Polynomial.aeval x) f)
          hf : Eq f 0
          ⊢ LT.lt f.natDegree d
        -/
        /-
          case pos
          R : Type u_1
          S : Type u_2
          inst✝² : CommRing R
          inst✝¹ : Ring S
          inst✝ : Algebra R S
          x y : S
          d : Nat
          hd : Ne d 0
          f : Polynomial R
          hy : Eq y ((Polynomial.aeval x) f)
          hf : Eq f 0
          h : LT.lt Bot.bot ↑d
          ⊢ LT.lt 0 d
        -/
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        x y : S
        d : Nat
        hd : Ne d 0
        f : Polynomial R
        h : LT.lt f.degree ↑d
        hy : Eq y ((Polynomial.aeval x) f)
        hf : Not (Eq f 0)
        ⊢ LT.lt f.natDegree d
      -/
        /-
          case neg
          R : Type u_1
          S : Type u_2
          inst✝² : CommRing R
          inst✝¹ : Ring S
          inst✝ : Algebra R S
          x y : S
          d : Nat
          hd : Ne d 0
          f : Polynomial R
          h : LT.lt ↑f.natDegree ↑d
          hy : Eq y ((Polynomial.aeval x) f)
          hf : Not (Eq f 0)
          ⊢ LT.lt f.natDegree d
        -/
        /-
          🎉 no goals
        -/
      · simp only [hf, natDegree_zero, degree_zero] at h ⊢
        /-
          case pos
          R : Type u_1
          S : Type u_2
          inst✝² : CommRing R
          inst✝¹ : Ring S
          inst✝ : Algebra R S
          x y : S
          d : Nat
          hd : Ne d 0
          f : Polynomial R
          hy : Eq y ((Polynomial.aeval x) f)
          hf : Eq f 0
          h : LT.lt 0 d
          ⊢ LT.lt Bot.bot ↑d
        -/
        first | exact lt_of_le_of_ne (Nat.zero_le d) hd.symm | exact WithBot.bot_lt_coe d
        /-
          🎉 no goals
        -/
      /-
        case neg
        R : Type u_1
        S : Type u_2
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        x y : S
        d : Nat
        hd : Ne d 0
        f : Polynomial R
        h : LT.lt f.natDegree d
        hy : Eq y ((Polynomial.aeval x) f)
        hf : Not (Eq f 0)
        ⊢ LT.lt f.degree ↑d
      -/
      simp_all only [degree_eq_natDegree hf]
        /-
          case neg
          R : Type u_1
          S : Type u_2
          inst✝² : CommRing R
          inst✝¹ : Ring S
          inst✝ : Algebra R S
          x y : S
          d : Nat
          hd : Ne d 0
          f : Polynomial R
          h : LT.lt f.natDegree d
          hy : Eq y ((Polynomial.aeval x) f)
          hf : Not (Eq f 0)
          ⊢ LT.lt ↑f.natDegree ↑d
        -/
      · first | exact WithBot.coe_lt_coe.1 h | exact WithBot.coe_lt_coe.2 h
        /-
          🎉 no goals
        -/


theorem dim_ne_zero [Nontrivial S] (pb : PowerBasis R S) : pb.dim ≠ 0 := fun h =>
  not_nonempty_iff.mpr (h.symm ▸ Fin.isEmpty : IsEmpty (Fin pb.dim)) pb.basis.index_nonempty


theorem dim_pos [Nontrivial S] (pb : PowerBasis R S) : 0 < pb.dim :=
  Nat.pos_of_ne_zero pb.dim_ne_zero


theorem exists_eq_aeval [Nontrivial S] (pb : PowerBasis R S) (y : S) :
    ∃ f : R[X], f.natDegree < pb.dim ∧ y = aeval pb.gen f :=
                                       /-
                                         R : Type u_1
                                         S : Type u_2
                                         inst✝³ : CommRing R
                                         inst✝² : Ring S
                                         inst✝¹ : Algebra R S
                                         inst✝ : Nontrivial S
                                         pb : PowerBasis R S
                                         y : S
                                         ⊢ Membership.mem (Submodule.span R (Set.range fun i => HPow.hPow pb.gen ↑i)) y
                                       -/
  (mem_span_pow pb.dim_ne_zero).mp (by simpa using pb.basis.mem_span y)
                                       /-
                                         🎉 no goals
                                       -/


theorem exists_eq_aeval' (pb : PowerBasis R S) (y : S) : ∃ f : R[X], y = aeval pb.gen f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    pb : PowerBasis R S
    y : S
    ⊢ Exists fun f => Eq y ((Polynomial.aeval pb.gen) f)
  -/
  nontriviality S
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    pb : PowerBasis R S
    y : S
    a✝ : Nontrivial S
    ⊢ Exists fun f => Eq y ((Polynomial.aeval pb.gen) f)
  -/
  obtain ⟨f, _, hf⟩ := exists_eq_aeval pb y
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    pb : PowerBasis R S
    y : S
    a✝ : Nontrivial S
    f : Polynomial R
    left✝ : LT.lt f.natDegree pb.dim
    hf : Eq y ((Polynomial.aeval pb.gen) f)
    ⊢ Exists fun f => Eq y ((Polynomial.aeval pb.gen) f)
  -/
  exact ⟨f, hf⟩
  /-
    🎉 no goals
  -/


theorem algHom_ext {S' : Type*} [Semiring S'] [Algebra R S'] (pb : PowerBasis R S)
    ⦃f g : S →ₐ[R] S'⦄ (h : f pb.gen = g pb.gen) : f = g := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    S' : Type u_7
    inst✝¹ : Semiring S'
    inst✝ : Algebra R S'
    pb : PowerBasis R S
    f g : AlgHom R S S'
    h : Eq (f pb.gen) (g pb.gen)
    ⊢ Eq f g
  -/
  ext x
  /-
    case H
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    S' : Type u_7
    inst✝¹ : Semiring S'
    inst✝ : Algebra R S'
    pb : PowerBasis R S
    f g : AlgHom R S S'
    h : Eq (f pb.gen) (g pb.gen)
    x : S
    ⊢ Eq (f x) (g x)
  -/
  obtain ⟨f, rfl⟩ := pb.exists_eq_aeval' x
  /-
    case H.intro
    R : Type u_1
    S : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    S' : Type u_7
    inst✝¹ : Semiring S'
    inst✝ : Algebra R S'
    pb : PowerBasis R S
    f✝ g : AlgHom R S S'
    h : Eq (f✝ pb.gen) (g pb.gen)
    f : Polynomial R
    ⊢ Eq (f✝ ((Polynomial.aeval pb.gen) f)) (g ((Polynomial.aeval pb.gen) f))
  -/
  rw [← Polynomial.aeval_algHom_apply, ← Polynomial.aeval_algHom_apply, h]
  /-
    🎉 no goals
  -/


open Ideal Finset Submodule in
theorem exists_smodEq (pb : PowerBasis A B) (b : B) :
    ∃ a, SModEq (Ideal.span ({pb.gen})) b (algebraMap A B a) := by
  /-
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    ⊢ Exists fun a => SModEq (Ideal.span (Singleton.singleton pb.gen)) b ((algebra …
  -/
  rcases subsingleton_or_nontrivial B
    /-
      case inl
      A : Type u_4
      B : Type u_5
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      pb : PowerBasis A B
      b : B
      h✝ : Subsingleton B
      ⊢ Exists fun a => SModEq (Ideal.span (Singleton.singleton pb.gen)) b ((algebra …
    -/
  · exact ⟨0, by rw [SModEq, Subsingleton.eq_zero b, _root_.map_zero]⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    ⊢ Exists fun a => SModEq (Ideal.span (Singleton.singleton pb.gen)) b ((algebra …
  -/
  refine ⟨pb.basis.repr b ⟨0, pb.dim_pos⟩, ?_⟩
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    ⊢ SModEq (Ideal.span (Singleton.singleton pb.gen)) b ((algebraMap A B) ((pb.ba …
  -/
  have H := pb.basis.sum_repr b
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    H : Eq (Finset.univ.sum fun i => HSMul.hSMul ((pb.basis.repr b) i) (pb.basis i …
    ⊢ SModEq (Ideal.span (Singleton.singleton pb.gen)) b ((algebraMap A B) ((pb.ba …
  -/
  rw [← insert_erase (mem_univ ⟨0, pb.dim_pos⟩), sum_insert (not_mem_erase _ _)] at H
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
    ⊢ SModEq (Ideal.span (Singleton.singleton pb.gen)) b ((algebraMap A B) ((pb.ba …
  -/
  rw [SModEq, ← add_zero (algebraMap _ _ _), Quotient.mk_add]
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
    ⊢ Eq (Submodule.Quotient.mk b) (HAdd.hAdd (Submodule.Quotient.mk ((algebraMap  …
  -/
  nth_rewrite 1 [← H]
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
    ⊢ Eq (Submodule.Quotient.mk (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) …
  -/
  rw [Quotient.mk_add]
  /-
    case inr
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    h✝ : Nontrivial B
    H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
    ⊢ Eq (HAdd.hAdd (Submodule.Quotient.mk (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) …
  -/
  congr 1
    /-
      case inr.e_a
      A : Type u_4
      B : Type u_5
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      pb : PowerBasis A B
      b : B
      h✝ : Nontrivial B
      H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
      ⊢ Eq (Submodule.Quotient.mk (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis  …
    -/
  · simp [Algebra.algebraMap_eq_smul_one ((pb.basis.repr b) _)]
    /-
      🎉 no goals
    -/
    /-
      case inr.e_a
      A : Type u_4
      B : Type u_5
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      pb : PowerBasis A B
      b : B
      h✝ : Nontrivial B
      H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
      ⊢ Eq (Submodule.Quotient.mk ((Finset.univ.erase ⟨0, ⋯⟩).sum fun x => HSMul.hSM …
    -/
  · rw [Quotient.mk_zero, Quotient.mk_eq_zero, coe_basis]
    /-
      case inr.e_a
      A : Type u_4
      B : Type u_5
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      pb : PowerBasis A B
      b : B
      h✝ : Nontrivial B
      H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
      ⊢ Membership.mem (Ideal.span (Singleton.singleton pb.gen)) ((Finset.univ.erase …
    -/
    refine sum_mem _ (fun i hi ↦ ?_)
    /-
      case inr.e_a
      A : Type u_4
      B : Type u_5
      inst✝² : CommRing A
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      pb : PowerBasis A B
      b : B
      h✝ : Nontrivial B
      H : Eq (HAdd.hAdd (HSMul.hSMul ((pb.basis.repr b) ⟨0, ⋯⟩) (pb.basis ⟨0, ⋯⟩)) ( …
      i : Fin pb.dim
      hi : Membership.mem (Finset.univ.erase ⟨0, ⋯⟩) i
      ⊢ Membership.mem (Ideal.span (Singleton.singleton pb.gen)) (HSMul.hSMul ((pb.b …
    -/
    rw [Algebra.smul_def']
    refine Ideal.mul_mem_left _ _ <| Ideal.pow_mem_of_mem _ (Ideal.subset_span (by simp)) _ <|
      Nat.pos_of_ne_zero <| fun h ↦ not_mem_erase i univ <| Fin.eq_mk_iff_val_eq.2 h ▸ hi


open Submodule.Quotient in
theorem exists_gen_dvd_sub (pb : PowerBasis A B) (b : B) : ∃ a, pb.gen ∣ b - algebraMap A B a := by
  /-
    A : Type u_4
    B : Type u_5
    inst✝² : CommRing A
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    pb : PowerBasis A B
    b : B
    ⊢ Exists fun a => Dvd.dvd pb.gen (HSub.hSub b ((algebraMap A B) a))
  -/
  simpa [← Ideal.mem_span_singleton, ← mk_eq_zero, mk_sub, sub_eq_zero] using pb.exists_smodEq b
  /-
    🎉 no goals
  -/


/-- `pb.minpolyGen` is the minimal polynomial for `pb.gen`. -/
noncomputable def minpolyGen (pb : PowerBasis A S) : A[X] :=
  X ^ pb.dim - ∑ i : Fin pb.dim, C (pb.basis.repr (pb.gen ^ pb.dim) i) * X ^ (i : ℕ)


theorem aeval_minpolyGen (pb : PowerBasis A S) : aeval pb.gen (minpolyGen pb) = 0 := by
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    ⊢ Eq ((Polynomial.aeval pb.gen) pb.minpolyGen) 0
  -/
  simp_rw [minpolyGen, map_sub, map_sum, map_mul, map_pow, aeval_C, ← Algebra.smul_def, aeval_X]
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    ⊢ Eq (HSub.hSub (HPow.hPow pb.gen pb.dim) (Finset.univ.sum fun x => HSMul.hSMu …
  -/
  refine sub_eq_zero.mpr ((pb.basis.linearCombination_repr (pb.gen ^ pb.dim)).symm.trans ?_)
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    ⊢ Eq ((Finsupp.linearCombination A ⇑pb.basis) (pb.basis.repr (HPow.hPow pb.gen …
  -/
  rw [Finsupp.linearCombination_apply, Finsupp.sum_fintype] <;>
    /-
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul ((pb.basis.repr (HPow.hPow pb.gen p …
    -/
    /-
      🎉 no goals
    -/
    simp only [pb.coe_basis, zero_smul, eq_self_iff_true, imp_true_iff]
    /-
      🎉 no goals
    -/


theorem minpolyGen_monic (pb : PowerBasis A S) : Monic (minpolyGen pb) := by
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    ⊢ pb.minpolyGen.Monic
  -/
  nontriviality A
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    a✝ : Nontrivial A
    ⊢ pb.minpolyGen.Monic
  -/
  apply (monic_X_pow _).sub_of_left _
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    a✝ : Nontrivial A
    ⊢ LT.lt (Finset.univ.sum fun i => HMul.hMul (Polynomial.C ((pb.basis.repr (HPo …
  -/
  rw [degree_X_pow]
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    a✝ : Nontrivial A
    ⊢ LT.lt (Finset.univ.sum fun i => HMul.hMul (Polynomial.C ((pb.basis.repr (HPo …
  -/
  exact degree_sum_fin_lt _
  /-
    🎉 no goals
  -/


theorem dim_le_natDegree_of_root (pb : PowerBasis A S) {p : A[X]} (ne_zero : p ≠ 0)
    (root : aeval pb.gen p = 0) : pb.dim ≤ p.natDegree := by
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    root : Eq ((Polynomial.aeval pb.gen) p) 0
    ⊢ LE.le pb.dim p.natDegree
  -/
  refine le_of_not_lt fun hlt => ne_zero ?_
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    root : Eq ((Polynomial.aeval pb.gen) p) 0
    hlt : LT.lt p.natDegree pb.dim
    ⊢ Eq p 0
  -/
  rw [p.as_sum_range' _ hlt, Finset.sum_range]
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    root : Eq ((Polynomial.aeval pb.gen) p) 0
    hlt : LT.lt p.natDegree pb.dim
    ⊢ Eq (Finset.univ.sum fun i => (Polynomial.monomial ↑i) (p.coeff ↑i)) 0
  -/
  refine Fintype.sum_eq_zero _ fun i => ?_
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    root : Eq ((Polynomial.aeval pb.gen) p) 0
    hlt : LT.lt p.natDegree pb.dim
    i : Fin pb.dim
    ⊢ Eq ((Polynomial.monomial ↑i) (p.coeff ↑i)) 0
  -/
  simp_rw [aeval_eq_sum_range' hlt, Finset.sum_range, ← pb.basis_eq_pow] at root
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    hlt : LT.lt p.natDegree pb.dim
    i : Fin pb.dim
    root : Eq (Finset.univ.sum fun x => HSMul.hSMul (p.coeff ↑x) (pb.basis x)) 0
    ⊢ Eq ((Polynomial.monomial ↑i) (p.coeff ↑i)) 0
  -/
  have := Fintype.linearIndependent_iff.1 pb.basis.linearIndependent _ root
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    hlt : LT.lt p.natDegree pb.dim
    i : Fin pb.dim
    root : Eq (Finset.univ.sum fun x => HSMul.hSMul (p.coeff ↑x) (pb.basis x)) 0
    this : ∀ (i : Fin pb.dim), Eq (p.coeff ↑i) 0
    ⊢ Eq ((Polynomial.monomial ↑i) (p.coeff ↑i)) 0
  -/
  rw [this, monomial_zero_right]
  /-
    🎉 no goals
  -/


theorem dim_le_degree_of_root (h : PowerBasis A S) {p : A[X]} (ne_zero : p ≠ 0)
    (root : aeval h.gen p = 0) : ↑h.dim ≤ p.degree := by
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    h : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    root : Eq ((Polynomial.aeval h.gen) p) 0
    ⊢ LE.le (↑h.dim) p.degree
  -/
  rw [degree_eq_natDegree ne_zero]
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    h : PowerBasis A S
    p : Polynomial A
    ne_zero : Ne p 0
    root : Eq ((Polynomial.aeval h.gen) p) 0
    ⊢ LE.le ↑h.dim ↑p.natDegree
  -/
  exact WithBot.coe_le_coe.2 (h.dim_le_natDegree_of_root ne_zero root)
  /-
    🎉 no goals
  -/


theorem degree_minpolyGen [Nontrivial A] (pb : PowerBasis A S) :
    degree (minpolyGen pb) = pb.dim := by
  /-
    S : Type u_2
    inst✝³ : Ring S
    A : Type u_4
    inst✝² : CommRing A
    inst✝¹ : Algebra A S
    inst✝ : Nontrivial A
    pb : PowerBasis A S
    ⊢ Eq pb.minpolyGen.degree ↑pb.dim
  -/
  unfold minpolyGen
  /-
    S : Type u_2
    inst✝³ : Ring S
    A : Type u_4
    inst✝² : CommRing A
    inst✝¹ : Algebra A S
    inst✝ : Nontrivial A
    pb : PowerBasis A S
    ⊢ Eq (HSub.hSub (HPow.hPow Polynomial.X pb.dim) (Finset.univ.sum fun i => HMul …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  rw [degree_sub_eq_left_of_degree_lt] <;> rw [degree_X_pow]
  /-
    S : Type u_2
    inst✝³ : Ring S
    A : Type u_4
    inst✝² : CommRing A
    inst✝¹ : Algebra A S
    inst✝ : Nontrivial A
    pb : PowerBasis A S
    ⊢ LT.lt (Finset.univ.sum fun i => HMul.hMul (Polynomial.C ((pb.basis.repr (HPo …
  -/
  apply degree_sum_fin_lt
  /-
    🎉 no goals
  -/


theorem natDegree_minpolyGen [Nontrivial A] (pb : PowerBasis A S) :
    natDegree (minpolyGen pb) = pb.dim :=
  natDegree_eq_of_degree_eq_some pb.degree_minpolyGen


@[simp]
theorem minpolyGen_eq (pb : PowerBasis A S) : pb.minpolyGen = minpoly A pb.gen := by
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    ⊢ Eq pb.minpolyGen (minpoly A pb.gen)
  -/
  nontriviality A
  refine minpoly.unique' A _ pb.minpolyGen_monic pb.aeval_minpolyGen fun q hq =>
    or_iff_not_imp_left.2 fun hn0 h0 => ?_
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    a✝ : Nontrivial A
    q : Polynomial A
    hq : LT.lt q.degree pb.minpolyGen.degree
    hn0 : Not (Eq q 0)
    h0 : Eq ((Polynomial.aeval pb.gen) q) 0
    ⊢ False
  -/
  exact (pb.dim_le_degree_of_root hn0 h0).not_lt (pb.degree_minpolyGen ▸ hq)
  /-
    🎉 no goals
  -/


theorem isIntegral_gen (pb : PowerBasis A S) : IsIntegral A pb.gen :=
  ⟨minpolyGen pb, minpolyGen_monic pb, aeval_minpolyGen pb⟩


@[simp]
theorem degree_minpoly [Nontrivial A] (pb : PowerBasis A S) :
                                             /-
                                               S : Type u_2
                                               inst✝³ : Ring S
                                               A : Type u_4
                                               inst✝² : CommRing A
                                               inst✝¹ : Algebra A S
                                               inst✝ : Nontrivial A
                                               pb : PowerBasis A S
                                               ⊢ Eq (minpoly A pb.gen).degree ↑pb.dim
                                             -/
    degree (minpoly A pb.gen) = pb.dim := by rw [← minpolyGen_eq, degree_minpolyGen]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem natDegree_minpoly [Nontrivial A] (pb : PowerBasis A S) :
                                                /-
                                                  S : Type u_2
                                                  inst✝³ : Ring S
                                                  A : Type u_4
                                                  inst✝² : CommRing A
                                                  inst✝¹ : Algebra A S
                                                  inst✝ : Nontrivial A
                                                  pb : PowerBasis A S
                                                  ⊢ Eq (minpoly A pb.gen).natDegree pb.dim
                                                -/
    (minpoly A pb.gen).natDegree = pb.dim := by rw [← minpolyGen_eq, natDegree_minpolyGen]
                                                /-
                                                  🎉 no goals
                                                -/


protected theorem leftMulMatrix (pb : PowerBasis A S) : Algebra.leftMulMatrix pb.basis pb.gen =
    @Matrix.of (Fin pb.dim) (Fin pb.dim) _ fun i j =>
      if ↑j + 1 = pb.dim then -pb.minpolyGen.coeff ↑i else if (i : ℕ) = j + 1 then 1 else 0 := by
  /-
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    ⊢ Eq ((Algebra.leftMulMatrix pb.basis) pb.gen) (Matrix.of fun i j => ite (Eq ( …
  -/
  cases subsingleton_or_nontrivial A; · subsingleton
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    h✝ : Nontrivial A
    ⊢ Eq ((Algebra.leftMulMatrix pb.basis) pb.gen) (Matrix.of fun i j => ite (Eq ( …
  -/
  rw [Algebra.leftMulMatrix_apply, ← LinearEquiv.eq_symm_apply, LinearMap.toMatrix_symm]
  /-
    case inr
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    h✝ : Nontrivial A
    ⊢ Eq ((Algebra.lmul A S) pb.gen) ((Matrix.toLin pb.basis pb.basis) (Matrix.of  …
  -/
  refine pb.basis.ext fun k => ?_
  /-
    case inr
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    h✝ : Nontrivial A
    k : Fin pb.dim
    ⊢ Eq (((Algebra.lmul A S) pb.gen) (pb.basis k)) (((Matrix.toLin pb.basis pb.ba …
  -/
  simp_rw [Matrix.toLin_self, Matrix.of_apply, pb.basis_eq_pow]
  /-
    case inr
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    h✝ : Nontrivial A
    k : Fin pb.dim
    ⊢ Eq (((Algebra.lmul A S) pb.gen) (HPow.hPow pb.gen ↑k)) (Finset.univ.sum fun  …
  -/
  apply (pow_succ' _ _).symm.trans
  /-
    case inr
    S : Type u_2
    inst✝² : Ring S
    A : Type u_4
    inst✝¹ : CommRing A
    inst✝ : Algebra A S
    pb : PowerBasis A S
    h✝ : Nontrivial A
    k : Fin pb.dim
    ⊢ Eq (HPow.hPow pb.gen (HAdd.hAdd (↑k) 1)) (Finset.univ.sum fun x => HSMul.hSM …
  -/
  split_ifs with h
    /-
      case pos
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      h✝ : Nontrivial A
      k : Fin pb.dim
      h : Eq (HAdd.hAdd (↑k) 1) pb.dim
      ⊢ Eq (HPow.hPow pb.gen (HAdd.hAdd (↑k) 1)) (Finset.univ.sum fun x => HSMul.hSM …
    -/
  · simp_rw [h, neg_smul, Finset.sum_neg_distrib, eq_neg_iff_add_eq_zero]
    /-
      case pos
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      h✝ : Nontrivial A
      k : Fin pb.dim
      h : Eq (HAdd.hAdd (↑k) 1) pb.dim
      ⊢ Eq (HAdd.hAdd (HPow.hPow pb.gen pb.dim) (Finset.univ.sum fun x => HSMul.hSMu …
    -/
    convert pb.aeval_minpolyGen
    rw [add_comm, aeval_eq_sum_range, Finset.sum_range_succ, ← leadingCoeff,
      pb.minpolyGen_monic.leadingCoeff, one_smul, natDegree_minpolyGen, Finset.sum_range]
    /-
      case neg
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      h✝ : Nontrivial A
      k : Fin pb.dim
      h : Not (Eq (HAdd.hAdd (↑k) 1) pb.dim)
      ⊢ Eq (HPow.hPow pb.gen (HAdd.hAdd (↑k) 1)) (Finset.univ.sum fun x => HSMul.hSM …
    -/
  · rw [Fintype.sum_eq_single (⟨(k : ℕ) + 1, lt_of_le_of_ne k.2 h⟩ : Fin pb.dim), if_pos, one_smul]
      /-
        case neg.hc
        S : Type u_2
        inst✝² : Ring S
        A : Type u_4
        inst✝¹ : CommRing A
        inst✝ : Algebra A S
        pb : PowerBasis A S
        h✝ : Nontrivial A
        k : Fin pb.dim
        h : Not (Eq (HAdd.hAdd (↑k) 1) pb.dim)
        ⊢ Eq (↑⟨HAdd.hAdd (↑k) 1, ⋯⟩) (HAdd.hAdd (↑k) 1)
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case neg
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      h✝ : Nontrivial A
      k : Fin pb.dim
      h : Not (Eq (HAdd.hAdd (↑k) 1) pb.dim)
      ⊢ ∀ (x : Fin pb.dim), Ne x ⟨HAdd.hAdd (↑k) 1, ⋯⟩ → Eq (HSMul.hSMul (ite (Eq (↑ …
    -/
    intro x hx
    /-
      case neg
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      h✝ : Nontrivial A
      k : Fin pb.dim
      h : Not (Eq (HAdd.hAdd (↑k) 1) pb.dim)
      x : Fin pb.dim
      hx : Ne x ⟨HAdd.hAdd (↑k) 1, ⋯⟩
      ⊢ Eq (HSMul.hSMul (ite (Eq (↑x) (HAdd.hAdd (↑k) 1)) 1 0) (HPow.hPow pb.gen ↑x) …
    -/
    rw [if_neg, zero_smul]
    /-
      case neg.hnc
      S : Type u_2
      inst✝² : Ring S
      A : Type u_4
      inst✝¹ : CommRing A
      inst✝ : Algebra A S
      pb : PowerBasis A S
      h✝ : Nontrivial A
      k : Fin pb.dim
      h : Not (Eq (HAdd.hAdd (↑k) 1) pb.dim)
      x : Fin pb.dim
      hx : Ne x ⟨HAdd.hAdd (↑k) 1, ⋯⟩
      ⊢ Not (Eq (↑x) (HAdd.hAdd (↑k) 1))
    -/
    apply mt Fin.ext hx
    /-
      🎉 no goals
    -/


theorem constr_pow_aeval (pb : PowerBasis A S) {y : S'} (hy : aeval y (minpoly A pb.gen) = 0)
    (f : A[X]) : pb.basis.constr A (fun i => y ^ (i : ℕ)) (aeval pb.gen f) = aeval y f := by
  /-
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) ((Polynomial.aeval pb.gen) …
  -/
  cases subsingleton_or_nontrivial A
    /-
      case inl
      S : Type u_2
      inst✝⁴ : Ring S
      A : Type u_4
      inst✝³ : CommRing A
      inst✝² : Algebra A S
      S' : Type u_7
      inst✝¹ : Ring S'
      inst✝ : Algebra A S'
      pb : PowerBasis A S
      y : S'
      hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
      f : Polynomial A
      h✝ : Subsingleton A
      ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) ((Polynomial.aeval pb.gen) …
    -/
  · rw [(Subsingleton.elim _ _ : f = 0), aeval_zero, map_zero, aeval_zero]
    /-
      🎉 no goals
    -/
  rw [← aeval_modByMonic_eq_self_of_root (minpoly.monic pb.isIntegral_gen) (minpoly.aeval _ _), ←
    @aeval_modByMonic_eq_self_of_root _ _ _ _ _ f _ (minpoly.monic pb.isIntegral_gen) y hy]
  /-
    case inr
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) ((Polynomial.aeval pb.gen) …
  -/
  by_cases hf : f %ₘ minpoly A pb.gen = 0
    /-
      case pos
      S : Type u_2
      inst✝⁴ : Ring S
      A : Type u_4
      inst✝³ : CommRing A
      inst✝² : Algebra A S
      S' : Type u_7
      inst✝¹ : Ring S'
      inst✝ : Algebra A S'
      pb : PowerBasis A S
      y : S'
      hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
      f : Polynomial A
      h✝ : Nontrivial A
      hf : Eq (f.modByMonic (minpoly A pb.gen)) 0
      ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) ((Polynomial.aeval pb.gen) …
    -/
  · simp only [hf, map_zero]
    /-
      🎉 no goals
    -/
  have : (f %ₘ minpoly A pb.gen).natDegree < pb.dim := by
    rw [← pb.natDegree_minpoly]
    apply natDegree_lt_natDegree hf
    exact degree_modByMonic_lt _ (minpoly.monic pb.isIntegral_gen)
  /-
    case neg
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    hf : Not (Eq (f.modByMonic (minpoly A pb.gen)) 0)
    this : LT.lt (f.modByMonic (minpoly A pb.gen)).natDegree pb.dim
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) ((Polynomial.aeval pb.gen) …
  -/
  rw [aeval_eq_sum_range' this, aeval_eq_sum_range' this, map_sum]
  /-
    case neg
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    hf : Not (Eq (f.modByMonic (minpoly A pb.gen)) 0)
    this : LT.lt (f.modByMonic (minpoly A pb.gen)).natDegree pb.dim
    ⊢ Eq ((Finset.range pb.dim).sum fun x => ((pb.basis.constr A) fun i => HPow.hP …
  -/
  refine Finset.sum_congr rfl fun i (hi : i ∈ Finset.range pb.dim) => ?_
  /-
    case neg
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    hf : Not (Eq (f.modByMonic (minpoly A pb.gen)) 0)
    this : LT.lt (f.modByMonic (minpoly A pb.gen)).natDegree pb.dim
    i : Nat
    hi : Membership.mem (Finset.range pb.dim) i
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) (HSMul.hSMul ((f.modByMoni …
  -/
  rw [Finset.mem_range] at hi
  /-
    case neg
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    hf : Not (Eq (f.modByMonic (minpoly A pb.gen)) 0)
    this : LT.lt (f.modByMonic (minpoly A pb.gen)).natDegree pb.dim
    i : Nat
    hi : LT.lt i pb.dim
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) (HSMul.hSMul ((f.modByMoni …
  -/
  rw [LinearMap.map_smul]
  /-
    case neg
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    hf : Not (Eq (f.modByMonic (minpoly A pb.gen)) 0)
    this : LT.lt (f.modByMonic (minpoly A pb.gen)).natDegree pb.dim
    i : Nat
    hi : LT.lt i pb.dim
    ⊢ Eq (HSMul.hSMul ((f.modByMonic (minpoly A pb.gen)).coeff i) (((pb.basis.cons …
  -/
  congr
  /-
    case neg.e_a
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f : Polynomial A
    h✝ : Nontrivial A
    hf : Not (Eq (f.modByMonic (minpoly A pb.gen)) 0)
    this : LT.lt (f.modByMonic (minpoly A pb.gen)).natDegree pb.dim
    i : Nat
    hi : LT.lt i pb.dim
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) (HPow.hPow pb.gen i)) (HPo …
  -/
  rw [← Fin.val_mk hi, ← pb.basis_eq_pow ⟨i, hi⟩, Basis.constr_basis]
  /-
    🎉 no goals
  -/


theorem constr_pow_gen (pb : PowerBasis A S) {y : S'} (hy : aeval y (minpoly A pb.gen) = 0) :
    pb.basis.constr A (fun i => y ^ (i : ℕ)) pb.gen = y := by
  /-
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) pb.gen) y
  -/
                                       /-
                                         🎉 no goals
                                       -/
  convert pb.constr_pow_aeval hy X <;> rw [aeval_X]
                                       /-
                                         🎉 no goals
                                       -/


theorem constr_pow_algebraMap (pb : PowerBasis A S) {y : S'} (hy : aeval y (minpoly A pb.gen) = 0)
    (x : A) : pb.basis.constr A (fun i => y ^ (i : ℕ)) (algebraMap A S x) = algebraMap A S' x := by
  /-
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    x : A
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) ((algebraMap A S) x)) ((al …
  -/
                                           /-
                                             🎉 no goals
                                           -/
  convert pb.constr_pow_aeval hy (C x) <;> rw [aeval_C]
                                           /-
                                             🎉 no goals
                                           -/


theorem constr_pow_mul (pb : PowerBasis A S) {y : S'} (hy : aeval y (minpoly A pb.gen) = 0)
    (x x' : S) : pb.basis.constr A (fun i => y ^ (i : ℕ)) (x * x') =
      pb.basis.constr A (fun i => y ^ (i : ℕ)) x * pb.basis.constr A (fun i => y ^ (i : ℕ)) x' := by
  /-
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    x x' : S
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) (HMul.hMul x x')) (HMul.hM …
  -/
  obtain ⟨f, rfl⟩ := pb.exists_eq_aeval' x
  /-
    case intro
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    x' : S
    f : Polynomial A
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) (HMul.hMul ((Polynomial.ae …
  -/
  obtain ⟨g, rfl⟩ := pb.exists_eq_aeval' x'
  /-
    case intro.intro
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    inst✝² : Algebra A S
    S' : Type u_7
    inst✝¹ : Ring S'
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    y : S'
    hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
    f g : Polynomial A
    ⊢ Eq (((pb.basis.constr A) fun i => HPow.hPow y ↑i) (HMul.hMul ((Polynomial.ae …
  -/
  simp only [← aeval_mul, pb.constr_pow_aeval hy]
  /-
    🎉 no goals
  -/


/-- `pb.lift y hy` is the algebra map sending `pb.gen` to `y`,
where `hy` states the higher powers of `y` are the same as the higher powers of `pb.gen`.

See `PowerBasis.liftEquiv` for a bundled equiv sending `⟨y, hy⟩` to the algebra map.
-/
noncomputable def lift (pb : PowerBasis A S) (y : S') (hy : aeval y (minpoly A pb.gen) = 0) :
    S →ₐ[A] S' :=
  { pb.basis.constr A fun i => y ^ (i : ℕ) with
                   /-
                     R : Type u_1
                     S : Type u_2
                     T : Type u_3
                     inst✝⁹ : CommRing R
                     inst✝⁸ : Ring S
                     inst✝⁷ : Algebra R S
                     A : Type u_4
                     B : Type u_5
                     inst✝⁶ : CommRing A
                     inst✝⁵ : CommRing B
                     inst✝⁴ : Algebra A B
                     K : Type u_6
                     inst✝³ : Field K
                     inst✝² : Algebra A S
                     S' : Type u_7
                     inst✝¹ : Ring S'
                     inst✝ : Algebra A S'
                     pb : PowerBasis A S
                     y : S'
                     hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
                     ⊢ Eq (__src✝.toFun 1) 1
                   -/
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    map_one' := by convert pb.constr_pow_algebraMap hy 1 using 2 <;> rw [RingHom.map_one]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      T : Type u_3
                      inst✝⁹ : CommRing R
                      inst✝⁸ : Ring S
                      inst✝⁷ : Algebra R S
                      A : Type u_4
                      B : Type u_5
                      inst✝⁶ : CommRing A
                      inst✝⁵ : CommRing B
                      inst✝⁴ : Algebra A B
                      K : Type u_6
                      inst✝³ : Field K
                      inst✝² : Algebra A S
                      S' : Type u_7
                      inst✝¹ : Ring S'
                      inst✝ : Algebra A S'
                      pb : PowerBasis A S
                      y : S'
                      hy : Eq ((Polynomial.aeval y) (minpoly A pb.gen)) 0
                      ⊢ Eq ((↑{ toFun := __src✝.toFun, map_one' := ⋯, map_mul' := ⋯ }).toFun 0) 0
                    -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    map_zero' := by convert pb.constr_pow_algebraMap hy 0 using 2 <;> rw [RingHom.map_zero]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    map_mul' := pb.constr_pow_mul hy
    commutes' := pb.constr_pow_algebraMap hy }


@[simp]
theorem lift_gen (pb : PowerBasis A S) (y : S') (hy : aeval y (minpoly A pb.gen) = 0) :
    pb.lift y hy pb.gen = y :=
  pb.constr_pow_gen hy


@[simp]
theorem lift_aeval (pb : PowerBasis A S) (y : S') (hy : aeval y (minpoly A pb.gen) = 0) (f : A[X]) :
    pb.lift y hy (aeval pb.gen f) = aeval y f :=
  pb.constr_pow_aeval hy f


/-- `pb.liftEquiv` states that roots of the minimal polynomial of `pb.gen` correspond to
maps sending `pb.gen` to that root.

This is the bundled equiv version of `PowerBasis.lift`.
If the codomain of the `AlgHom`s is an integral domain, then the roots form a multiset,
see `liftEquiv'` for the corresponding statement.
-/
@[simps]
noncomputable def liftEquiv (pb : PowerBasis A S) :
    (S →ₐ[A] S') ≃ { y : S' // aeval y (minpoly A pb.gen) = 0 } where
                           /-
                             R : Type u_1
                             S : Type u_2
                             T : Type u_3
                             inst✝⁹ : CommRing R
                             inst✝⁸ : Ring S
                             inst✝⁷ : Algebra R S
                             A : Type u_4
                             B : Type u_5
                             inst✝⁶ : CommRing A
                             inst✝⁵ : CommRing B
                             inst✝⁴ : Algebra A B
                             K : Type u_6
                             inst✝³ : Field K
                             inst✝² : Algebra A S
                             S' : Type u_7
                             inst✝¹ : Ring S'
                             inst✝ : Algebra A S'
                             pb : PowerBasis A S
                             f : AlgHom A S S'
                             ⊢ Eq ((Polynomial.aeval (f pb.gen)) (minpoly A pb.gen)) 0
                           -/
  toFun f := ⟨f pb.gen, by rw [aeval_algHom_apply, minpoly.aeval, map_zero]⟩
                           /-
                             🎉 no goals
                           -/
  invFun y := pb.lift y y.2
  left_inv _ := pb.algHom_ext <| lift_gen _ _ _
  right_inv y := Subtype.ext <| lift_gen _ _ y.prop


/-- `pb.liftEquiv'` states that elements of the root set of the minimal
polynomial of `pb.gen` correspond to maps sending `pb.gen` to that root. -/
@[simps! (config := .asFn)]
noncomputable def liftEquiv' [IsDomain B] (pb : PowerBasis A S) :
    (S →ₐ[A] B) ≃ { y : B // y ∈ (minpoly A pb.gen).aroots B } :=
  pb.liftEquiv.trans ((Equiv.refl _).subtypeEquiv fun x => by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝¹⁰ : CommRing R
      inst✝⁹ : Ring S
      inst✝⁸ : Algebra R S
      A : Type u_4
      B : Type u_5
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra A B
      K : Type u_6
      inst✝⁴ : Field K
      inst✝³ : Algebra A S
      S' : Type u_7
      inst✝² : Ring S'
      inst✝¹ : Algebra A S'
      inst✝ : IsDomain B
      pb : PowerBasis A S
      x : B
      ⊢ Iff (Eq ((Polynomial.aeval x) (minpoly A pb.gen)) 0) (Membership.mem ((minpo …
    -/
    rw [Equiv.refl_apply, mem_roots_iff_aeval_eq_zero]
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝¹⁰ : CommRing R
        inst✝⁹ : Ring S
        inst✝⁸ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁷ : CommRing A
        inst✝⁶ : CommRing B
        inst✝⁵ : Algebra A B
        K : Type u_6
        inst✝⁴ : Field K
        inst✝³ : Algebra A S
        S' : Type u_7
        inst✝² : Ring S'
        inst✝¹ : Algebra A S'
        inst✝ : IsDomain B
        pb : PowerBasis A S
        x : B
        ⊢ Iff (Eq ((Polynomial.aeval x) (minpoly A pb.gen)) 0) (Eq ((Polynomial.aeval  …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝¹⁰ : CommRing R
        inst✝⁹ : Ring S
        inst✝⁸ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁷ : CommRing A
        inst✝⁶ : CommRing B
        inst✝⁵ : Algebra A B
        K : Type u_6
        inst✝⁴ : Field K
        inst✝³ : Algebra A S
        S' : Type u_7
        inst✝² : Ring S'
        inst✝¹ : Algebra A S'
        inst✝ : IsDomain B
        pb : PowerBasis A S
        x : B
        ⊢ Ne (Polynomial.map (algebraMap A B) (minpoly A pb.gen)) 0
      -/
    · exact map_monic_ne_zero (minpoly.monic pb.isIntegral_gen))
      /-
        🎉 no goals
      -/


/-- There are finitely many algebra homomorphisms `S →ₐ[A] B` if `S` is of the form `A[x]`
and `B` is an integral domain. -/
noncomputable def AlgHom.fintype [IsDomain B] (pb : PowerBasis A S) : Fintype (S →ₐ[A] B) :=
  letI := Classical.decEq B
  Fintype.ofEquiv _ pb.liftEquiv'.symm


/-- `pb.equivOfRoot pb' h₁ h₂` is an equivalence of algebras with the same power basis,
where "the same" means that `pb` is a root of `pb'`s minimal polynomial and vice versa.

See also `PowerBasis.equivOfMinpoly` which takes the hypothesis that the
minimal polynomials are identical.
-/
@[simps! (config := .lemmasOnly) apply]
noncomputable def equivOfRoot (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h₁ : aeval pb.gen (minpoly A pb'.gen) = 0) (h₂ : aeval pb'.gen (minpoly A pb.gen) = 0) :
    S ≃ₐ[A] S' :=
  AlgEquiv.ofAlgHom (pb.lift pb'.gen h₂) (pb'.lift pb.gen h₁)
    (by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring S
        inst✝⁷ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        K : Type u_6
        inst✝³ : Field K
        inst✝² : Algebra A S
        S' : Type u_7
        inst✝¹ : Ring S'
        inst✝ : Algebra A S'
        pb : PowerBasis A S
        pb' : PowerBasis A S'
        h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A pb'.gen)) 0
        h₂ : Eq ((Polynomial.aeval pb'.gen) (minpoly A pb.gen)) 0
        ⊢ Eq ((pb.lift pb'.gen h₂).comp (pb'.lift pb.gen h₁)) (AlgHom.id A S')
      -/
      ext x
      /-
        case H
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring S
        inst✝⁷ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        K : Type u_6
        inst✝³ : Field K
        inst✝² : Algebra A S
        S' : Type u_7
        inst✝¹ : Ring S'
        inst✝ : Algebra A S'
        pb : PowerBasis A S
        pb' : PowerBasis A S'
        h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A pb'.gen)) 0
        h₂ : Eq ((Polynomial.aeval pb'.gen) (minpoly A pb.gen)) 0
        x : S'
        ⊢ Eq (((pb.lift pb'.gen h₂).comp (pb'.lift pb.gen h₁)) x) ((AlgHom.id A S') x)
      -/
      obtain ⟨f, hf, rfl⟩ := pb'.exists_eq_aeval' x
      /-
        case H.intro.refl
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring S
        inst✝⁷ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        K : Type u_6
        inst✝³ : Field K
        inst✝² : Algebra A S
        S' : Type u_7
        inst✝¹ : Ring S'
        inst✝ : Algebra A S'
        pb : PowerBasis A S
        pb' : PowerBasis A S'
        h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A pb'.gen)) 0
        h₂ : Eq ((Polynomial.aeval pb'.gen) (minpoly A pb.gen)) 0
        f : Polynomial A
        ⊢ Eq (((pb.lift pb'.gen h₂).comp (pb'.lift pb.gen h₁)) ((Polynomial.aeval pb'. …
      -/
      simp)
      /-
        🎉 no goals
      -/
    (by
      /-
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring S
        inst✝⁷ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        K : Type u_6
        inst✝³ : Field K
        inst✝² : Algebra A S
        S' : Type u_7
        inst✝¹ : Ring S'
        inst✝ : Algebra A S'
        pb : PowerBasis A S
        pb' : PowerBasis A S'
        h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A pb'.gen)) 0
        h₂ : Eq ((Polynomial.aeval pb'.gen) (minpoly A pb.gen)) 0
        ⊢ Eq ((pb'.lift pb.gen h₁).comp (pb.lift pb'.gen h₂)) (AlgHom.id A S)
      -/
      ext x
      /-
        case H
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring S
        inst✝⁷ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        K : Type u_6
        inst✝³ : Field K
        inst✝² : Algebra A S
        S' : Type u_7
        inst✝¹ : Ring S'
        inst✝ : Algebra A S'
        pb : PowerBasis A S
        pb' : PowerBasis A S'
        h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A pb'.gen)) 0
        h₂ : Eq ((Polynomial.aeval pb'.gen) (minpoly A pb.gen)) 0
        x : S
        ⊢ Eq (((pb'.lift pb.gen h₁).comp (pb.lift pb'.gen h₂)) x) ((AlgHom.id A S) x)
      -/
      obtain ⟨f, hf, rfl⟩ := pb.exists_eq_aeval' x
      /-
        case H.intro.refl
        R : Type u_1
        S : Type u_2
        T : Type u_3
        inst✝⁹ : CommRing R
        inst✝⁸ : Ring S
        inst✝⁷ : Algebra R S
        A : Type u_4
        B : Type u_5
        inst✝⁶ : CommRing A
        inst✝⁵ : CommRing B
        inst✝⁴ : Algebra A B
        K : Type u_6
        inst✝³ : Field K
        inst✝² : Algebra A S
        S' : Type u_7
        inst✝¹ : Ring S'
        inst✝ : Algebra A S'
        pb : PowerBasis A S
        pb' : PowerBasis A S'
        h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A pb'.gen)) 0
        h₂ : Eq ((Polynomial.aeval pb'.gen) (minpoly A pb.gen)) 0
        f : Polynomial A
        ⊢ Eq (((pb'.lift pb.gen h₁).comp (pb.lift pb'.gen h₂)) ((Polynomial.aeval pb.g …
      -/
      simp)
      /-
        🎉 no goals
      -/


@[simp]
theorem equivOfRoot_aeval (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h₁ : aeval pb.gen (minpoly A pb'.gen) = 0) (h₂ : aeval pb'.gen (minpoly A pb.gen) = 0)
    (f : A[X]) : pb.equivOfRoot pb' h₁ h₂ (aeval pb.gen f) = aeval pb'.gen f :=
  pb.lift_aeval _ h₂ _


@[simp]
theorem equivOfRoot_gen (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h₁ : aeval pb.gen (minpoly A pb'.gen) = 0) (h₂ : aeval pb'.gen (minpoly A pb.gen) = 0) :
    pb.equivOfRoot pb' h₁ h₂ pb.gen = pb'.gen :=
  pb.lift_gen _ h₂


@[simp]
theorem equivOfRoot_symm (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h₁ : aeval pb.gen (minpoly A pb'.gen) = 0) (h₂ : aeval pb'.gen (minpoly A pb.gen) = 0) :
    (pb.equivOfRoot pb' h₁ h₂).symm = pb'.equivOfRoot pb h₂ h₁ :=
  rfl


/-- `pb.equivOfMinpoly pb' h` is an equivalence of algebras with the same power basis,
where "the same" means that they have identical minimal polynomials.

See also `PowerBasis.equivOfRoot` which takes the hypothesis that each generator is a root of the
other basis' minimal polynomial; `PowerBasis.equivOfRoot` is more general if `A` is not a field.
-/
@[simps! (config := .lemmasOnly) apply]
noncomputable def equivOfMinpoly (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h : minpoly A pb.gen = minpoly A pb'.gen) : S ≃ₐ[A] S' :=
  pb.equivOfRoot pb' (h ▸ minpoly.aeval _ _) (h.symm ▸ minpoly.aeval _ _)


@[simp]
theorem equivOfMinpoly_aeval (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h : minpoly A pb.gen = minpoly A pb'.gen) (f : A[X]) :
    pb.equivOfMinpoly pb' h (aeval pb.gen f) = aeval pb'.gen f :=
  pb.equivOfRoot_aeval pb' _ _ _


@[simp]
theorem equivOfMinpoly_gen (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h : minpoly A pb.gen = minpoly A pb'.gen) : pb.equivOfMinpoly pb' h pb.gen = pb'.gen :=
  pb.equivOfRoot_gen pb' _ _


@[simp]
theorem equivOfMinpoly_symm (pb : PowerBasis A S) (pb' : PowerBasis A S')
    (h : minpoly A pb.gen = minpoly A pb'.gen) :
    (pb.equivOfMinpoly pb' h).symm = pb'.equivOfMinpoly pb h.symm :=
  rfl


/-- Useful lemma to show `x` generates a power basis:
the powers of `x` less than the degree of `x`'s minimal polynomial are linearly independent. -/
theorem linearIndependent_pow [Algebra K S] (x : S) :
    LinearIndependent K fun i : Fin (minpoly K x).natDegree => x ^ (i : ℕ) := by
  /-
    S : Type u_2
    inst✝² : Ring S
    K : Type u_6
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x : S
    ⊢ LinearIndependent K fun i => HPow.hPow x ↑i
  -/
  by_cases h : IsIntegral K x; swap
    /-
      case neg
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : Not (IsIntegral K x)
      ⊢ LinearIndependent K fun i => HPow.hPow x ↑i
    -/
  · rw [minpoly.eq_zero h, natDegree_zero]
    /-
      case neg
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : Not (IsIntegral K x)
      ⊢ LinearIndependent K fun i => HPow.hPow x ↑i
    -/
    exact linearIndependent_empty_type
    /-
      🎉 no goals
    -/
  /-
    case pos
    S : Type u_2
    inst✝² : Ring S
    K : Type u_6
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x : S
    h : IsIntegral K x
    ⊢ LinearIndependent K fun i => HPow.hPow x ↑i
  -/
  refine Fintype.linearIndependent_iff.2 fun g hg i => ?_
  /-
    case pos
    S : Type u_2
    inst✝² : Ring S
    K : Type u_6
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x : S
    h : IsIntegral K x
    g : Fin (minpoly K x).natDegree → K
    hg : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (HPow.hPow x ↑i)) 0
    i : Fin (minpoly K x).natDegree
    ⊢ Eq (g i) 0
  -/
  simp only at hg
  /-
    case pos
    S : Type u_2
    inst✝² : Ring S
    K : Type u_6
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x : S
    h : IsIntegral K x
    g : Fin (minpoly K x).natDegree → K
    hg : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (HPow.hPow x ↑i)) 0
    i : Fin (minpoly K x).natDegree
    ⊢ Eq (g i) 0
  -/
  simp_rw [Algebra.smul_def, ← aeval_monomial, ← map_sum] at hg
  /-
    case pos
    S : Type u_2
    inst✝² : Ring S
    K : Type u_6
    inst✝¹ : Field K
    inst✝ : Algebra K S
    x : S
    h : IsIntegral K x
    g : Fin (minpoly K x).natDegree → K
    i : Fin (minpoly K x).natDegree
    hg : Eq ((Polynomial.aeval x) (Finset.univ.sum fun x_1 => (Polynomial.monomial …
    ⊢ Eq (g i) 0
  -/
  apply (fun hn0 => (minpoly.degree_le_of_ne_zero K x (mt (fun h0 => ?_) hn0) hg).not_lt).mtr
    /-
      case pos
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : IsIntegral K x
      g : Fin (minpoly K x).natDegree → K
      i : Fin (minpoly K x).natDegree
      hg : Eq ((Polynomial.aeval x) (Finset.univ.sum fun x_1 => (Polynomial.monomial …
      ⊢ LT.lt (Finset.univ.sum fun x_1 => (Polynomial.monomial ↑x_1) (g x_1)).degree …
    -/
  · simp_rw [← C_mul_X_pow_eq_monomial]
    /-
      case pos
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : IsIntegral K x
      g : Fin (minpoly K x).natDegree → K
      i : Fin (minpoly K x).natDegree
      hg : Eq ((Polynomial.aeval x) (Finset.univ.sum fun x_1 => (Polynomial.monomial …
      ⊢ LT.lt (Finset.univ.sum fun x_1 => HMul.hMul (Polynomial.C (g x_1)) (HPow.hPo …
    -/
    exact (degree_eq_natDegree <| minpoly.ne_zero h).symm ▸ degree_sum_fin_lt _
    /-
      🎉 no goals
    -/
    /-
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : IsIntegral K x
      g : Fin (minpoly K x).natDegree → K
      i : Fin (minpoly K x).natDegree
      hg : Eq ((Polynomial.aeval x) (Finset.univ.sum fun x_1 => (Polynomial.monomial …
      hn0 : Not (Eq (g i) 0)
      h0 : Eq (Finset.univ.sum fun x_1 => (Polynomial.monomial ↑x_1) (g x_1)) 0
      ⊢ Eq (g i) 0
    -/
  · apply_fun lcoeff K i at h0
    /-
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : IsIntegral K x
      g : Fin (minpoly K x).natDegree → K
      i : Fin (minpoly K x).natDegree
      hg : Eq ((Polynomial.aeval x) (Finset.univ.sum fun x_1 => (Polynomial.monomial …
      hn0 : Not (Eq (g i) 0)
      h0 : Eq ((Polynomial.lcoeff K ↑i) (Finset.univ.sum fun x_1 => (Polynomial.mono …
      ⊢ Eq (g i) 0
    -/
    simp_rw [map_sum, lcoeff_apply, coeff_monomial, Fin.val_eq_val, Finset.sum_ite_eq'] at h0
    /-
      S : Type u_2
      inst✝² : Ring S
      K : Type u_6
      inst✝¹ : Field K
      inst✝ : Algebra K S
      x : S
      h : IsIntegral K x
      g : Fin (minpoly K x).natDegree → K
      i : Fin (minpoly K x).natDegree
      hg : Eq ((Polynomial.aeval x) (Finset.univ.sum fun x_1 => (Polynomial.monomial …
      hn0 : Not (Eq (g i) 0)
      h0 : Eq (ite (Membership.mem Finset.univ i) (g i) 0) (Polynomial.coeff 0 ↑i)
      ⊢ Eq (g i) 0
    -/
    exact (if_pos <| Finset.mem_univ _).symm.trans h0
    /-
      🎉 no goals
    -/


theorem IsIntegral.mem_span_pow [Nontrivial R] {x y : S} (hx : IsIntegral R x)
    (hy : ∃ f : R[X], y = aeval x f) :
    y ∈ Submodule.span R (Set.range fun i : Fin (minpoly R x).natDegree => x ^ (i : ℕ)) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial R
    x y : S
    hx : IsIntegral R x
    hy : Exists fun f => Eq y ((Polynomial.aeval x) f)
    ⊢ Membership.mem (Submodule.span R (Set.range fun i => HPow.hPow x ↑i)) y
  -/
  obtain ⟨f, rfl⟩ := hy
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial R
    x : S
    hx : IsIntegral R x
    f : Polynomial R
    ⊢ Membership.mem (Submodule.span R (Set.range fun i => HPow.hPow x ↑i)) ((Poly …
  -/
  apply mem_span_pow'.mpr _
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial R
    x : S
    hx : IsIntegral R x
    f : Polynomial R
    ⊢ Exists fun f_1 => And (LT.lt f_1.degree ↑(minpoly R x).natDegree) (Eq ((Poly …
  -/
  have := minpoly.monic hx
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial R
    x : S
    hx : IsIntegral R x
    f : Polynomial R
    this : (minpoly R x).Monic
    ⊢ Exists fun f_1 => And (LT.lt f_1.degree ↑(minpoly R x).natDegree) (Eq ((Poly …
  -/
  refine ⟨f %ₘ minpoly R x, (degree_modByMonic_lt _ this).trans_le degree_le_natDegree, ?_⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial R
    x : S
    hx : IsIntegral R x
    f : Polynomial R
    this : (minpoly R x).Monic
    ⊢ Eq ((Polynomial.aeval x) f) ((Polynomial.aeval x) (f.modByMonic (minpoly R x …
  -/
  conv_lhs => rw [← modByMonic_add_div f this]
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    inst✝ : Nontrivial R
    x : S
    hx : IsIntegral R x
    f : Polynomial R
    this : (minpoly R x).Monic
    ⊢ Eq ((Polynomial.aeval x) (HAdd.hAdd (f.modByMonic (minpoly R x)) (HMul.hMul  …
  -/
  simp only [add_zero, zero_mul, minpoly.aeval, aeval_add, map_mul]
  /-
    🎉 no goals
  -/


/-- `PowerBasis.map pb (e : S ≃ₐ[R] S')` is the power basis for `S'` generated by `e pb.gen`. -/
@[simps dim gen basis]
noncomputable def map (pb : PowerBasis R S) (e : S ≃ₐ[R] S') : PowerBasis R S' where
  dim := pb.dim
  basis := pb.basis.map e.toLinearEquiv
  gen := e pb.gen
                       /-
                         R : Type u_1
                         S : Type u_2
                         T : Type u_3
                         inst✝⁸ : CommRing R
                         inst✝⁷ : Ring S
                         inst✝⁶ : Algebra R S
                         A : Type u_4
                         B : Type u_5
                         inst✝⁵ : CommRing A
                         inst✝⁴ : CommRing B
                         inst✝³ : Algebra A B
                         K : Type u_6
                         inst✝² : Field K
                         S' : Type u_7
                         inst✝¹ : CommRing S'
                         inst✝ : Algebra R S'
                         pb : PowerBasis R S
                         e : AlgEquiv R S S'
                         i : Fin pb.dim
                         ⊢ Eq ((pb.basis.map e.toLinearEquiv) i) (HPow.hPow (e pb.gen) ↑i)
                       -/
  basis_eq_pow i := by rw [Basis.map_apply, pb.basis_eq_pow, e.toLinearEquiv_apply, map_pow]
                       /-
                         🎉 no goals
                       -/


theorem minpolyGen_map (pb : PowerBasis A S) (e : S ≃ₐ[A] S') :
    (pb.map e).minpolyGen = pb.minpolyGen := by
  /-
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    S' : Type u_7
    inst✝² : CommRing S'
    inst✝¹ : Algebra A S
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    e : AlgEquiv A S S'
    ⊢ Eq (pb.map e).minpolyGen pb.minpolyGen
  -/
  dsimp only [minpolyGen, map_dim]
  -- Turn `Fin (pb.map e).dim` into `Fin pb.dim`
  simp only [LinearEquiv.trans_apply, map_basis, Basis.map_repr, map_gen,
    AlgEquiv.toLinearEquiv_apply, e.toLinearEquiv_symm, map_pow,
    AlgEquiv.symm_apply_apply, sub_right_inj]


@[simp]
theorem equivOfRoot_map (pb : PowerBasis A S) (e : S ≃ₐ[A] S') (h₁ h₂) :
    pb.equivOfRoot (pb.map e) h₁ h₂ = e := by
  /-
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    S' : Type u_7
    inst✝² : CommRing S'
    inst✝¹ : Algebra A S
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    e : AlgEquiv A S S'
    h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A (pb.map e).gen)) 0
    h₂ : Eq ((Polynomial.aeval (pb.map e).gen) (minpoly A pb.gen)) 0
    ⊢ Eq (pb.equivOfRoot (pb.map e) h₁ h₂) e
  -/
  ext x
  /-
    case h
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    S' : Type u_7
    inst✝² : CommRing S'
    inst✝¹ : Algebra A S
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    e : AlgEquiv A S S'
    h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A (pb.map e).gen)) 0
    h₂ : Eq ((Polynomial.aeval (pb.map e).gen) (minpoly A pb.gen)) 0
    x : S
    ⊢ Eq ((pb.equivOfRoot (pb.map e) h₁ h₂) x) (e x)
  -/
  obtain ⟨f, rfl⟩ := pb.exists_eq_aeval' x
  /-
    case h.intro
    S : Type u_2
    inst✝⁴ : Ring S
    A : Type u_4
    inst✝³ : CommRing A
    S' : Type u_7
    inst✝² : CommRing S'
    inst✝¹ : Algebra A S
    inst✝ : Algebra A S'
    pb : PowerBasis A S
    e : AlgEquiv A S S'
    h₁ : Eq ((Polynomial.aeval pb.gen) (minpoly A (pb.map e).gen)) 0
    h₂ : Eq ((Polynomial.aeval (pb.map e).gen) (minpoly A pb.gen)) 0
    f : Polynomial A
    ⊢ Eq ((pb.equivOfRoot (pb.map e) h₁ h₂) ((Polynomial.aeval pb.gen) f)) (e ((Po …
  -/
  simp [aeval_algEquiv]
  /-
    🎉 no goals
  -/


@[simp]
theorem equivOfMinpoly_map (pb : PowerBasis A S) (e : S ≃ₐ[A] S')
    (h : minpoly A pb.gen = minpoly A (pb.map e).gen) : pb.equivOfMinpoly (pb.map e) h = e :=
  pb.equivOfRoot_map _ _ _


theorem adjoin_gen_eq_top (B : PowerBasis R S) : adjoin R ({B.gen} : Set S) = ⊤ := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    ⊢ Eq (Algebra.adjoin R (Singleton.singleton B.gen)) Top.top
  -/
  rw [← toSubmodule_eq_top, _root_.eq_top_iff, ← B.basis.span_eq, Submodule.span_le]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    ⊢ HasSubset.Subset (Set.range ⇑B.basis) ↑(Subalgebra.toSubmodule (Algebra.adjo …
  -/
  rintro x ⟨i, rfl⟩
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    i : Fin B.dim
    ⊢ Membership.mem (↑(Subalgebra.toSubmodule (Algebra.adjoin R (Singleton.single …
  -/
  rw [B.basis_eq_pow i]
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    i : Fin B.dim
    ⊢ Membership.mem (↑(Subalgebra.toSubmodule (Algebra.adjoin R (Singleton.single …
  -/
  exact Subalgebra.pow_mem _ (subset_adjoin (Set.mem_singleton _)) _
  /-
    🎉 no goals
  -/


theorem adjoin_eq_top_of_gen_mem_adjoin {B : PowerBasis R S} {x : S}
    (hx : B.gen ∈ adjoin R ({x} : Set S)) : adjoin R ({x} : Set S) = ⊤ := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    x : S
    hx : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) B.gen
    ⊢ Eq (Algebra.adjoin R (Singleton.singleton x)) Top.top
  -/
  rw [_root_.eq_top_iff, ← B.adjoin_gen_eq_top]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    x : S
    hx : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) B.gen
    ⊢ LE.le (Algebra.adjoin R (Singleton.singleton B.gen)) (Algebra.adjoin R (Sing …
  -/
  refine adjoin_le ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    B : PowerBasis R S
    x : S
    hx : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) B.gen
    ⊢ HasSubset.Subset (Singleton.singleton B.gen) ↑(Algebra.adjoin R (Singleton.s …
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


