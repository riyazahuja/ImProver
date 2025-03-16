theorem linearIndependent_single {φ : ι → Type*} {f : ∀ ι, φ ι → M}
    (hf : ∀ i, LinearIndependent R (f i)) :
    LinearIndependent R fun ix : Σi, φ i => single ix.1 (f ix.1 ix.2) := by
  /-
    R : Type u_1
    M : Type u_2
    ι : Type u_3
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    φ : ι → Type u_4
    f : (ι : ι) → φ ι → M
    hf : ∀ (i : ι), LinearIndependent R (f i)
    ⊢ LinearIndependent R fun ix => Finsupp.single ix.fst (f ix.fst ix.snd)
  -/
  apply @linearIndependent_iUnion_finite R _ _ _ _ ι φ fun i x => single i (f i x)
    /-
      case hindep
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      φ : ι → Type u_4
      f : (ι : ι) → φ ι → M
      hf : ∀ (i : ι), LinearIndependent R (f i)
      ⊢ ∀ (j : ι), LinearIndependent R fun x => Finsupp.single j (f j x)
    -/
  · intro i
    have h_disjoint : Disjoint (span R (range (f i))) (ker (lsingle i)) := by
      rw [ker_lsingle]
      exact disjoint_bot_right
    /-
      case hindep
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      φ : ι → Type u_4
      f : (ι : ι) → φ ι → M
      hf : ∀ (i : ι), LinearIndependent R (f i)
      i : ι
      h_disjoint : Disjoint (Submodule.span R (Set.range (f i))) (LinearMap.ker (Fin …
      ⊢ LinearIndependent R fun x => Finsupp.single i (f i x)
    -/
    apply (hf i).map h_disjoint
    /-
      🎉 no goals
    -/
    /-
      case hd
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      φ : ι → Type u_4
      f : (ι : ι) → φ ι → M
      hf : ∀ (i : ι), LinearIndependent R (f i)
      ⊢ ∀ (i : ι) (t : Set ι), t.Finite → Not (Membership.mem t i) → Disjoint (Submo …
    -/
  · intro i t _ hit
    /-
      case hd
      R : Type u_1
      M : Type u_2
      ι : Type u_3
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      φ : ι → Type u_4
      f : (ι : ι) → φ ι → M
      hf : ∀ (i : ι), LinearIndependent R (f i)
      i : ι
      t : Set ι
      a✝ : t.Finite
      hit : Not (Membership.mem t i)
      ⊢ Disjoint (Submodule.span R (Set.range fun x => Finsupp.single i (f i x))) (i …
    -/
    refine (disjoint_lsingle_lsingle {i} t (disjoint_singleton_left.2 hit)).mono ?_ ?_
      /-
        case hd.refine_1
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i)
        ⊢ LE.le (Submodule.span R (Set.range fun x => Finsupp.single i (f i x))) (iSup …
      -/
    · rw [span_le]
      /-
        case hd.refine_1
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i)
        ⊢ HasSubset.Subset (Set.range fun x => Finsupp.single i (f i x)) ↑(iSup fun a  …
      -/
      simp only [iSup_singleton]
      /-
        case hd.refine_1
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i)
        ⊢ HasSubset.Subset (Set.range fun x => Finsupp.single i (f i x)) ↑(LinearMap.r …
      -/
      rw [range_coe]
      /-
        case hd.refine_1
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i)
        ⊢ HasSubset.Subset (Set.range fun x => Finsupp.single i (f i x)) (Set.range ⇑( …
      -/
      apply range_comp_subset_range _ (lsingle i)
      /-
        🎉 no goals
      -/
      /-
        case hd.refine_2
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i)
        ⊢ LE.le (iSup fun i => iSup fun h => Submodule.span R (Set.range fun x => Fins …
      -/
    · refine iSup₂_mono fun i hi => ?_
      /-
        case hd.refine_2
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i✝ : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i✝)
        i : ι
        hi : Membership.mem t i
        ⊢ LE.le (Submodule.span R (Set.range fun x => Finsupp.single i (f i x))) (Line …
      -/
      rw [span_le, range_coe]
      /-
        case hd.refine_2
        R : Type u_1
        M : Type u_2
        ι : Type u_3
        inst✝² : Ring R
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        φ : ι → Type u_4
        f : (ι : ι) → φ ι → M
        hf : ∀ (i : ι), LinearIndependent R (f i)
        i✝ : ι
        t : Set ι
        a✝ : t.Finite
        hit : Not (Membership.mem t i✝)
        i : ι
        hi : Membership.mem t i
        ⊢ HasSubset.Subset (Set.range fun x => Finsupp.single i (f i x)) (Set.range ⇑( …
      -/
      apply range_comp_subset_range _ (lsingle i)
      /-
        🎉 no goals
      -/


open scoped Classical in
/-- The basis on `ι →₀ M` with basis vectors `fun ⟨i, x⟩ ↦ single i (b i x)`. -/
protected def basis {φ : ι → Type*} (b : ∀ i, Basis (φ i) R M) : Basis (Σi, φ i) R (ι →₀ M) :=
  Basis.ofRepr
    { toFun := fun g =>
        { toFun := fun ix => (b ix.1).repr (g ix.1) ix.2
          support := g.support.sigma fun i => ((b i).repr (g i)).support
          mem_support_toFun := fun ix => by
            /-
              R : Type u_1
              M : Type u_2
              ι : Type u_3
              inst✝² : Semiring R
              inst✝¹ : AddCommMonoid M
              inst✝ : Module R M
              φ : ι → Type u_4
              b : (i : ι) → Basis (φ i) R M
              g : Finsupp ι M
              ix : Sigma fun i => φ i
              ⊢ Iff (Membership.mem (g.support.sigma fun i => ((b i).repr (g i)).support) ix …
            -/
            simp only [Finset.mem_sigma, mem_support_iff, and_iff_right_iff_imp, Ne]
            /-
              R : Type u_1
              M : Type u_2
              ι : Type u_3
              inst✝² : Semiring R
              inst✝¹ : AddCommMonoid M
              inst✝ : Module R M
              φ : ι → Type u_4
              b : (i : ι) → Basis (φ i) R M
              g : Finsupp ι M
              ix : Sigma fun i => φ i
              ⊢ Not (Eq (((b ix.fst).repr (g ix.fst)) ix.snd) 0) → Not (Eq (g ix.fst) 0)
            -/
            intro b hg
            /-
              R : Type u_1
              M : Type u_2
              ι : Type u_3
              inst✝² : Semiring R
              inst✝¹ : AddCommMonoid M
              inst✝ : Module R M
              φ : ι → Type u_4
              b✝ : (i : ι) → Basis (φ i) R M
              g : Finsupp ι M
              ix : Sigma fun i => φ i
              b : Not (Eq (((b✝ ix.fst).repr (g ix.fst)) ix.snd) 0)
              hg : Eq (g ix.fst) 0
              ⊢ False
            -/
            simp [hg] at b }
            /-
              🎉 no goals
            -/
      invFun := fun g =>
        { toFun := fun i => (b i).repr.symm (g.comapDomain _ sigma_mk_injective.injOn)
          support := g.support.image Sigma.fst
          mem_support_toFun := fun i => by
            rw [Ne, ← (b i).repr.injective.eq_iff, (b i).repr.apply_symm_apply,
                DFunLike.ext_iff]
            simp only [exists_prop, LinearEquiv.map_zero, comapDomain_apply, zero_apply,
              exists_and_right, mem_support_iff, exists_eq_right, Sigma.exists, Finset.mem_image,
              not_forall] }
      left_inv := fun g => by
        /-
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g : Finsupp ι M
          ⊢ Eq ((fun g => { support := Finset.image Sigma.fst g.support, toFun := fun i  …
        -/
        ext i
        /-
          case h
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g : Finsupp ι M
          i : ι
          ⊢ Eq (((fun g => { support := Finset.image Sigma.fst g.support, toFun := fun i …
        -/
        rw [← (b i).repr.injective.eq_iff]
        /-
          case h
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g : Finsupp ι M
          i : ι
          ⊢ Eq ((b i).repr (((fun g => { support := Finset.image Sigma.fst g.support, to …
        -/
        ext x
        /-
          case h.h
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g : Finsupp ι M
          i : ι
          x : φ i
          ⊢ Eq (((b i).repr (((fun g => { support := Finset.image Sigma.fst g.support, t …
        -/
        simp only [coe_mk, LinearEquiv.apply_symm_apply, comapDomain_apply]
        /-
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g h : Finsupp ι M
          ⊢ Eq ((fun g => { support := g.support.sigma fun i => ((b i).repr (g i)).suppo …
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.mk
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g h : Finsupp ι M
          i : ι
          x : φ i
          ⊢ Eq (((fun g => { support := g.support.sigma fun i => ((b i).repr (g i)).supp …
        -/
      right_inv := fun g => by
        /-
          🎉 no goals
        -/
        /-
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g : Finsupp (Sigma fun i => φ i) R
          ⊢ Eq ({ toFun := fun g => { support := g.support.sigma fun i => ((b i).repr (g …
        -/
        /-
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          c : R
          h : Finsupp ι M
          ⊢ Eq ({ toFun := fun g => { support := g.support.sigma fun i => ((b i).repr (g …
        -/
        ext ⟨i, x⟩
        /-
          case h.mk
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          c : R
          h : Finsupp ι M
          i : ι
          x : φ i
          ⊢ Eq (({ toFun := fun g => { support := g.support.sigma fun i => ((b i).repr ( …
        -/
        /-
          case h.mk
          R : Type u_1
          M : Type u_2
          ι : Type u_3
          inst✝² : Semiring R
          inst✝¹ : AddCommMonoid M
          inst✝ : Module R M
          φ : ι → Type u_4
          b : (i : ι) → Basis (φ i) R M
          g : Finsupp (Sigma fun i => φ i) R
          i : ι
          x : φ i
          ⊢ Eq (({ toFun := fun g => { support := g.support.sigma fun i => ((b i).repr ( …
        -/
        /-
          🎉 no goals
        -/
        simp only [coe_mk, LinearEquiv.apply_symm_apply, comapDomain_apply]
        /-
          🎉 no goals
        -/
      map_add' := fun g h => by
        ext ⟨i, x⟩
        simp only [coe_mk, add_apply, LinearEquiv.map_add]
      map_smul' := fun c h => by
        ext ⟨i, x⟩
        simp only [coe_mk, smul_apply, LinearEquiv.map_smul, RingHom.id_apply] }


@[simp]
theorem basis_repr {φ : ι → Type*} (b : ∀ i, Basis (φ i) R M) (g : ι →₀ M) (ix) :
    (Finsupp.basis b).repr g ix = (b ix.1).repr (g ix.1) ix.2 :=
  rfl


@[simp]
theorem coe_basis {φ : ι → Type*} (b : ∀ i, Basis (φ i) R M) :
    ⇑(Finsupp.basis b) = fun ix : Σi, φ i => single ix.1 (b ix.1 ix.2) :=
  funext fun ⟨i, x⟩ =>
    Basis.apply_eq_iff.mpr <| by
      classical
      ext ⟨j, y⟩
      by_cases h : i = j
      · cases h
        simp only [basis_repr, single_eq_same, Basis.repr_self,
          Finsupp.single_apply_left sigma_mk_injective]
      · have : Sigma.mk i x ≠ Sigma.mk j y := fun h' => h <| congrArg (fun s => s.fst) h'
        -- Porting note: previously `this` not needed
        simp only [basis_repr, single_apply, h, this, if_false, LinearEquiv.map_zero, zero_apply]


/-- The basis on `ι →₀ M` with basis vectors `fun i ↦ single i 1`. -/
@[simps]
protected def basisSingleOne : Basis ι R (ι →₀ R) :=
  Basis.ofRepr (LinearEquiv.refl _ _)


@[simp]
theorem coe_basisSingleOne : (Finsupp.basisSingleOne : ι → ι →₀ R) = fun i => Finsupp.single i 1 :=
  funext fun _ => Basis.apply_eq_iff.mpr rfl


/-- The direct sum of free modules is free.

Note that while this is stated for `DFinsupp` not `DirectSum`, the types are defeq. -/
noncomputable def basis {η : ι → Type*} (b : ∀ i, Basis (η i) R (M i)) :
    Basis (Σi, η i) R (Π₀ i, M i) :=
  .ofRepr
    ((mapRange.linearEquiv fun i => (b i).repr).trans (sigmaFinsuppLequivDFinsupp R).symm)


theorem _root_.Finset.sum_single_ite [Fintype n] (a : R) (i : n) :
    (∑ x : n, Finsupp.single x (if i = x then a else 0)) = Finsupp.single i a := by
  simp only [apply_ite (Finsupp.single _), Finsupp.single_zero, Finset.sum_ite_eq,
    if_pos (Finset.mem_univ _)]


@[simp]
theorem equivFun_symm_single [Finite n] (b : Basis n R M) (i : n) :
    b.equivFun.symm (Pi.single i 1) = b i := by
  /-
    R : Type u_1
    M : Type u_2
    n : Type u_3
    inst✝⁴ : DecidableEq n
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite n
    b : Basis n R M
    i : n
    ⊢ Eq (b.equivFun.symm (Pi.single i 1)) (b i)
  -/
  cases nonempty_fintype n
  /-
    case intro
    R : Type u_1
    M : Type u_2
    n : Type u_3
    inst✝⁴ : DecidableEq n
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : Finite n
    b : Basis n R M
    i : n
    val✝ : Fintype n
    ⊢ Eq (b.equivFun.symm (Pi.single i 1)) (b i)
  -/
  simp [Pi.single_apply]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated equivFun_symm_single (since := "2024-08-09")]
theorem equivFun_symm_stdBasis [Finite n] (b : Basis n R M) (i : n) :
    b.equivFun.symm (LinearMap.stdBasis R (fun _ => R) i 1) = b i :=
  equivFun_symm_single ..


