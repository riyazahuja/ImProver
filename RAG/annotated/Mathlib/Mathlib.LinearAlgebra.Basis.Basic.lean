theorem coe_sumCoords_eq_finsum : (b.sumCoords : M → R) = fun m => ∑ᶠ i, b.coord i m := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    b : Basis ι R M
    ⊢ Eq ⇑b.sumCoords fun m => finsum fun i => (b.coord i) m
  -/
  ext m
  simp only [Basis.sumCoords, Basis.coord, Finsupp.lapply_apply, LinearMap.id_coe,
    LinearEquiv.coe_coe, Function.comp_apply, Finsupp.coe_lsum, LinearMap.coe_comp,
    finsum_eq_sum _ (b.repr m).finite_support, Finsupp.sum, Finset.finite_toSet_toFinset, id,
    Finsupp.fun_support_eq]


protected theorem linearIndependent : LinearIndependent R b :=
  fun x y hxy => by
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_5
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      b : Basis ι R M
      x y : Finsupp ι R
      hxy : Eq ((Finsupp.linearCombination R ⇑b) x) ((Finsupp.linearCombination R ⇑b …
      ⊢ Eq x y
    -/
    rw [← b.repr_linearCombination x, hxy, b.repr_linearCombination y]
    /-
      🎉 no goals
    -/


protected theorem ne_zero [Nontrivial R] (i) : b i ≠ 0 :=
  b.linearIndependent.ne_zero i


/-- `Basis.prod` maps an `ι`-indexed basis for `M` and an `ι'`-indexed basis for `M'`
to an `ι ⊕ ι'`-index basis for `M × M'`.
For the specific case of `R × R`, see also `Basis.finTwoProd`. -/
protected def prod : Basis (ι ⊕ ι') R (M × M') :=
  ofRepr ((b.repr.prod b'.repr).trans (Finsupp.sumFinsuppLEquivProdFinsupp R).symm)


@[simp]
theorem prod_repr_inl (x) (i) : (b.prod b').repr x (Sum.inl i) = b.repr x.1 i :=
  rfl


@[simp]
theorem prod_repr_inr (x) (i) : (b.prod b').repr x (Sum.inr i) = b'.repr x.2 i :=
  rfl


theorem prod_apply_inl_fst (i) : (b.prod b' (Sum.inl i)).1 = b i :=
  b.repr.injective <| by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i : ι
      ⊢ Eq (b.repr ((b.prod b') (Sum.inl i)).1) (b.repr (b i))
    -/
    ext j
    simp only [Basis.prod, Basis.coe_ofRepr, LinearEquiv.symm_trans_apply, LinearEquiv.prod_symm,
      LinearEquiv.prod_apply, b.repr.apply_symm_apply, LinearEquiv.symm_symm, repr_self,
      Equiv.toFun_as_coe, Finsupp.fst_sumFinsuppLEquivProdFinsupp]
    /-
      case h
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i j : ι
      ⊢ Eq ((Finsupp.single (Sum.inl i) 1) (Sum.inl j)) ((Finsupp.single i 1) j)
    -/
    apply Finsupp.single_apply_left Sum.inl_injective
    /-
      🎉 no goals
    -/


theorem prod_apply_inr_fst (i) : (b.prod b' (Sum.inr i)).1 = 0 :=
  b.repr.injective <| by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i : ι'
      ⊢ Eq (b.repr ((b.prod b') (Sum.inr i)).1) (b.repr 0)
    -/
    ext i
    simp only [Basis.prod, Basis.coe_ofRepr, LinearEquiv.symm_trans_apply, LinearEquiv.prod_symm,
      LinearEquiv.prod_apply, b.repr.apply_symm_apply, LinearEquiv.symm_symm, repr_self,
      Equiv.toFun_as_coe, Finsupp.fst_sumFinsuppLEquivProdFinsupp, LinearEquiv.map_zero,
      Finsupp.zero_apply]
    /-
      case h
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i✝ : ι'
      i : ι
      ⊢ Eq ((Finsupp.single (Sum.inr i✝) 1) (Sum.inl i)) 0
    -/
    apply Finsupp.single_eq_of_ne Sum.inr_ne_inl
    /-
      🎉 no goals
    -/


theorem prod_apply_inl_snd (i) : (b.prod b' (Sum.inl i)).2 = 0 :=
  b'.repr.injective <| by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i : ι
      ⊢ Eq (b'.repr ((b.prod b') (Sum.inl i)).2) (b'.repr 0)
    -/
    ext j
    simp only [Basis.prod, Basis.coe_ofRepr, LinearEquiv.symm_trans_apply, LinearEquiv.prod_symm,
      LinearEquiv.prod_apply, b'.repr.apply_symm_apply, LinearEquiv.symm_symm, repr_self,
      Equiv.toFun_as_coe, Finsupp.snd_sumFinsuppLEquivProdFinsupp, LinearEquiv.map_zero,
      Finsupp.zero_apply]
    /-
      case h
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i : ι
      j : ι'
      ⊢ Eq ((Finsupp.single (Sum.inl i) 1) (Sum.inr j)) 0
    -/
    apply Finsupp.single_eq_of_ne Sum.inl_ne_inr
    /-
      🎉 no goals
    -/


theorem prod_apply_inr_snd (i) : (b.prod b' (Sum.inr i)).2 = b' i :=
  b'.repr.injective <| by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i : ι'
      ⊢ Eq (b'.repr ((b.prod b') (Sum.inr i)).2) (b'.repr (b' i))
    -/
    ext i
    simp only [Basis.prod, Basis.coe_ofRepr, LinearEquiv.symm_trans_apply, LinearEquiv.prod_symm,
      LinearEquiv.prod_apply, b'.repr.apply_symm_apply, LinearEquiv.symm_symm, repr_self,
      Equiv.toFun_as_coe, Finsupp.snd_sumFinsuppLEquivProdFinsupp]
    /-
      case h
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      M : Type u_5
      M' : Type u_6
      inst✝⁴ : Semiring R
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : AddCommMonoid M'
      inst✝ : Module R M'
      b : Basis ι R M
      b' : Basis ι' R M'
      i✝ i : ι'
      ⊢ Eq ((Finsupp.single (Sum.inr i✝) 1) (Sum.inr i)) ((Finsupp.single i✝ 1) i)
    -/
    apply Finsupp.single_apply_left Sum.inr_injective
    /-
      🎉 no goals
    -/


@[simp]
theorem prod_apply (i) :
    b.prod b' i = Sum.elim (LinearMap.inl R M M' ∘ b) (LinearMap.inr R M M' ∘ b') i := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    M : Type u_5
    M' : Type u_6
    inst✝⁴ : Semiring R
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : AddCommMonoid M'
    inst✝ : Module R M'
    b : Basis ι R M
    b' : Basis ι' R M'
    i : Sum ι ι'
    ⊢ Eq ((b.prod b') i) (Sum.elim (Function.comp ⇑(LinearMap.inl R M M') ⇑b) (Fun …
  -/
  ext <;> cases i <;>
    simp only [prod_apply_inl_fst, Sum.elim_inl, LinearMap.inl_apply, prod_apply_inr_fst,
      Sum.elim_inr, LinearMap.inr_apply, prod_apply_inl_snd, prod_apply_inr_snd, Function.comp]


protected theorem noZeroSMulDivisors [NoZeroDivisors R] (b : Basis ι R M) :
    NoZeroSMulDivisors R M :=
  ⟨fun {c x} hcx => by
    exact or_iff_not_imp_right.mpr fun hx => by
      rw [← b.linearCombination_repr x, ← LinearMap.map_smul,
        ← map_zero (linearCombination R b)] at hcx
      have := b.linearIndependent hcx
      rw [smul_eq_zero] at this
      exact this.resolve_right fun hr => hx (b.repr.map_eq_zero_iff.mp hr)⟩


protected theorem smul_eq_zero [NoZeroDivisors R] (b : Basis ι R M) {c : R} {x : M} :
    c • x = 0 ↔ c = 0 ∨ x = 0 :=
  @smul_eq_zero _ _ _ _ _ b.noZeroSMulDivisors _ _


theorem basis_singleton_iff {R M : Type*} [Ring R] [Nontrivial R] [AddCommGroup M] [Module R M]
    [NoZeroSMulDivisors R M] (ι : Type*) [Unique ι] :
    Nonempty (Basis ι R M) ↔ ∃ x ≠ 0, ∀ y : M, ∃ r : R, r • x = y := by
  /-
    R : Type u_7
    M : Type u_8
    inst✝⁵ : Ring R
    inst✝⁴ : Nontrivial R
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : NoZeroSMulDivisors R M
    ι : Type u_9
    inst✝ : Unique ι
    ⊢ Iff (Nonempty (Basis ι R M)) (Exists fun x => And (Ne x 0) (∀ (y : M), Exist …
  -/
  constructor
    /-
      case mp
      R : Type u_7
      M : Type u_8
      inst✝⁵ : Ring R
      inst✝⁴ : Nontrivial R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      ι : Type u_9
      inst✝ : Unique ι
      ⊢ Nonempty (Basis ι R M) → Exists fun x => And (Ne x 0) (∀ (y : M), Exists fun …
    -/
  · rintro ⟨b⟩
    /-
      case mp.intro
      R : Type u_7
      M : Type u_8
      inst✝⁵ : Ring R
      inst✝⁴ : Nontrivial R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      ι : Type u_9
      inst✝ : Unique ι
      b : Basis ι R M
      ⊢ Exists fun x => And (Ne x 0) (∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x …
    -/
    refine ⟨b default, b.linearIndependent.ne_zero _, ?_⟩
    /-
      case mp.intro
      R : Type u_7
      M : Type u_8
      inst✝⁵ : Ring R
      inst✝⁴ : Nontrivial R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      ι : Type u_9
      inst✝ : Unique ι
      b : Basis ι R M
      ⊢ ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r (b Inhabited.default)) y
    -/
    simpa [span_singleton_eq_top_iff, Set.range_unique] using b.span_eq
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_7
      M : Type u_8
      inst✝⁵ : Ring R
      inst✝⁴ : Nontrivial R
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : NoZeroSMulDivisors R M
      ι : Type u_9
      inst✝ : Unique ι
      ⊢ (Exists fun x => And (Ne x 0) (∀ (y : M), Exists fun r => Eq (HSMul.hSMul r  …
    -/
  · rintro ⟨x, nz, w⟩
    refine ⟨ofRepr <| LinearEquiv.symm
      { toFun := fun f => f default • x
        invFun := fun y => Finsupp.single default (w y).choose
        left_inv := fun f => Finsupp.unique_ext ?_
        right_inv := fun y => ?_
        map_add' := fun y z => ?_
        map_smul' := fun c y => ?_ }⟩
      /-
        case mpr.intro.intro.refine_1
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        y z : Finsupp ι R
        ⊢ Eq ((fun f => HSMul.hSMul (f Inhabited.default) x) (HAdd.hAdd y z)) (HAdd.hA …
      -/
    · simp [Finsupp.add_apply, add_smul]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.refine_2
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        c : R
        y : Finsupp ι R
        ⊢ Eq ({ toFun := fun f => HSMul.hSMul (f Inhabited.default) x, map_add' := ⋯ } …
      -/
    · simp only [Finsupp.coe_smul, Pi.smul_apply, RingHom.id_apply]
      /-
        case mpr.intro.intro.refine_2
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        c : R
        y : Finsupp ι R
        ⊢ Eq (HSMul.hSMul (HSMul.hSMul c (y Inhabited.default)) x) (HSMul.hSMul c (HSM …
      -/
      rw [← smul_assoc]
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.refine_3
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        f : Finsupp ι R
        ⊢ Eq (((fun y => Finsupp.single Inhabited.default ⋯.choose) ({ toFun := fun f  …
      -/
    · refine smul_left_injective _ nz ?_
      /-
        case mpr.intro.intro.refine_3
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        f : Finsupp ι R
        ⊢ Eq ((fun c => HSMul.hSMul c x) (((fun y => Finsupp.single Inhabited.default  …
      -/
      simp only [Finsupp.single_eq_same]
      /-
        case mpr.intro.intro.refine_3
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        f : Finsupp ι R
        ⊢ Eq (HSMul.hSMul ⋯.choose x) (HSMul.hSMul (f Inhabited.default) x)
      -/
      exact (w (f default • x)).choose_spec
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.refine_4
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        y : M
        ⊢ Eq ({ toFun := fun f => HSMul.hSMul (f Inhabited.default) x, map_add' := ⋯,  …
      -/
    · simp only [Finsupp.single_eq_same]
      /-
        case mpr.intro.intro.refine_4
        R : Type u_7
        M : Type u_8
        inst✝⁵ : Ring R
        inst✝⁴ : Nontrivial R
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : NoZeroSMulDivisors R M
        ι : Type u_9
        inst✝ : Unique ι
        x : M
        nz : Ne x 0
        w : ∀ (y : M), Exists fun r => Eq (HSMul.hSMul r x) y
        y : M
        ⊢ Eq (HSMul.hSMul ⋯.choose x) y
      -/
      exact (w y).choose_spec
      /-
        🎉 no goals
      -/


theorem Basis.eq_bot_of_rank_eq_zero [NoZeroDivisors R] (b : Basis ι R M) (N : Submodule R M)
    (rank_eq : ∀ {m : ℕ} (v : Fin m → N), LinearIndependent R ((↑) ∘ v : Fin m → M) → m = 0) :
    N = ⊥ := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    rank_eq : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    ⊢ Eq N Bot.bot
  -/
  rw [Submodule.eq_bot_iff]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    rank_eq : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    ⊢ ∀ (x : M), Membership.mem N x → Eq x 0
  -/
  intro x hx
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    rank_eq : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    x : M
    hx : Membership.mem N x
    ⊢ Eq x 0
  -/
  contrapose! rank_eq with x_ne
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    x : M
    hx : Membership.mem N x
    x_ne : Ne x 0
    ⊢ Exists fun {m} => Exists fun v => And (LinearIndependent R (Function.comp Su …
  -/
  refine ⟨1, fun _ => ⟨x, hx⟩, ?_, one_ne_zero⟩
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    x : M
    hx : Membership.mem N x
    x_ne : Ne x 0
    ⊢ LinearIndependent R (Function.comp Subtype.val fun x_1 => ⟨x, hx⟩)
  -/
  rw [Fintype.linearIndependent_iff]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    x : M
    hx : Membership.mem N x
    x_ne : Ne x 0
    ⊢ ∀ (g : Fin 1 → R), Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Function. …
  -/
  rintro g sum_eq i
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    x : M
    hx : Membership.mem N x
    x_ne : Ne x 0
    g : Fin 1 → R
    sum_eq : Eq (Finset.univ.sum fun i => HSMul.hSMul (g i) (Function.comp Subtype …
    i : Fin 1
    ⊢ Eq (g i) 0
  -/
  cases' i with _ hi
  simp only [Function.const_apply, Fin.default_eq_zero, Submodule.coe_mk, Finset.univ_unique,
    Function.comp_const, Finset.sum_singleton] at sum_eq
  /-
    case mk
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : NoZeroDivisors R
    b : Basis ι R M
    N : Submodule R M
    x : M
    hx : Membership.mem N x
    x_ne : Ne x 0
    g : Fin 1 → R
    val✝ : Nat
    hi : LT.lt val✝ 1
    sum_eq : Eq (HSMul.hSMul (g 0) (Function.comp Subtype.val (fun x_1 => ⟨x, hx⟩) …
    ⊢ Eq (g ⟨val✝, hi⟩) 0
  -/
  convert (b.smul_eq_zero.mp sum_eq).resolve_right x_ne
  /-
    🎉 no goals
  -/


/-- Any basis is a maximal linear independent set.
-/
theorem maximal [Nontrivial R] (b : Basis ι R M) : b.linearIndependent.Maximal := fun w hi h => by
  -- If `w` is strictly bigger than `range b`,
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    ⊢ Eq (Set.range ⇑b) w
  -/
  apply le_antisymm h
  -- then choose some `x ∈ w \ range b`,
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    ⊢ LE.le w (Set.range ⇑b)
  -/
  intro x p
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    ⊢ Membership.mem (Set.range ⇑b) x
  -/
  by_contra q
  -- and write it in terms of the basis.
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    ⊢ False
  -/
  have e := b.linearCombination_repr x
  -- This then expresses `x` as a linear combination
  -- of elements of `w` which are in the range of `b`,
  let u : ι ↪ w :=
    ⟨fun i => ⟨b i, h ⟨i, rfl⟩⟩, fun i i' r =>
      b.injective (by simpa only [Subtype.mk_eq_mk] using r)⟩
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    e : Eq ((Finsupp.linearCombination R ⇑b) (b.repr x)) x
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    ⊢ False
  -/
  simp_rw [Finsupp.linearCombination_apply] at e
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    e : Eq ((b.repr x).sum fun i a => HSMul.hSMul a (b i)) x
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    ⊢ False
  -/
  change ((b.repr x).sum fun (i : ι) (a : R) ↦ a • (u i : M)) = ((⟨x, p⟩ : w) : M) at e
  rw [← Finsupp.sum_embDomain (f := u) (g := fun x r ↦ r • (x : M)),
      ← Finsupp.linearCombination_apply] at e
  -- Now we can contradict the linear independence of `hi`
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    e : Eq ((Finsupp.linearCombination R Subtype.val) (Finsupp.embDomain u (b.repr …
    ⊢ False
  -/
  refine hi.linearCombination_ne_of_not_mem_support _ ?_ e
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    e : Eq ((Finsupp.linearCombination R Subtype.val) (Finsupp.embDomain u (b.repr …
    ⊢ Not (Membership.mem (Finsupp.embDomain u (b.repr x)).support ⟨x, p⟩)
  -/
  simp only [Finset.mem_map, Finsupp.support_embDomain]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    e : Eq ((Finsupp.linearCombination R Subtype.val) (Finsupp.embDomain u (b.repr …
    ⊢ Not (Exists fun a => And (Membership.mem (b.repr x).support a) (Eq (u a) ⟨x, …
  -/
  rintro ⟨j, -, W⟩
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    e : Eq ((Finsupp.linearCombination R Subtype.val) (Finsupp.embDomain u (b.repr …
    j : ι
    W : Eq (u j) ⟨x, p⟩
    ⊢ False
  -/
  simp only [u, Embedding.coeFn_mk, Subtype.mk_eq_mk] at W
  /-
    case intro.intro
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : Nontrivial R
    b : Basis ι R M
    w : Set M
    hi : LinearIndependent R Subtype.val
    h : LE.le (Set.range ⇑b) w
    x : M
    p : Membership.mem w x
    q : Not (Membership.mem (Set.range ⇑b) x)
    u : Function.Embedding ι ↑w := { toFun := fun i => ⟨b i, ⋯⟩, inj' := ⋯ }
    e : Eq ((Finsupp.linearCombination R Subtype.val) (Finsupp.embDomain u (b.repr …
    j : ι
    W : Eq (b j) x
    ⊢ False
  -/
  apply q ⟨j, W⟩
  /-
    🎉 no goals
  -/


/-- A linear independent family of vectors spanning the whole module is a basis. -/
protected noncomputable def mk : Basis ι R M :=
  .ofRepr
    { hli.repr.comp (LinearMap.id.codRestrict _ fun _ => hsp Submodule.mem_top) with
      invFun := Finsupp.linearCombination _ v
      left_inv := fun x => hli.linearCombination_repr ⟨x, _⟩
      right_inv := fun _ => hli.repr_eq rfl }


@[simp]
theorem mk_repr : (Basis.mk hli hsp).repr x = hli.repr ⟨x, hsp Submodule.mem_top⟩ :=
  rfl


theorem mk_apply (i : ι) : Basis.mk hli hsp i = v i :=
                                                /-
                                                  ι : Type u_1
                                                  R : Type u_3
                                                  M : Type u_5
                                                  v : ι → M
                                                  inst✝² : Ring R
                                                  inst✝¹ : AddCommGroup M
                                                  inst✝ : Module R M
                                                  hli : LinearIndependent R v
                                                  hsp : LE.le Top.top (Submodule.span R (Set.range v))
                                                  i : ι
                                                  ⊢ Eq ((Finsupp.linearCombination R v) (Finsupp.single i 1)) (v i)
                                                -/
  show Finsupp.linearCombination _ v _ = v i by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem coe_mk : ⇑(Basis.mk hli hsp) = v :=
  funext (mk_apply _ _)


/-- Given a basis, the `i`th element of the dual basis evaluates to 1 on the `i`th element of the
basis. -/
theorem mk_coord_apply_eq (i : ι) : (Basis.mk hli hsp).coord i (v i) = 1 :=
                                                                         /-
                                                                           ι : Type u_1
                                                                           R : Type u_3
                                                                           M : Type u_5
                                                                           v : ι → M
                                                                           inst✝² : Ring R
                                                                           inst✝¹ : AddCommGroup M
                                                                           inst✝ : Module R M
                                                                           hli : LinearIndependent R v
                                                                           hsp : LE.le Top.top (Submodule.span R (Set.range v))
                                                                           i : ι
                                                                           ⊢ Eq ((hli.repr ⟨v i, ⋯⟩) i) 1
                                                                         -/
  show hli.repr ⟨v i, Submodule.subset_span (mem_range_self i)⟩ i = 1 by simp [hli.repr_eq_single i]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- Given a basis, the `i`th element of the dual basis evaluates to 0 on the `j`th element of the
basis if `j ≠ i`. -/
theorem mk_coord_apply_ne {i j : ι} (h : j ≠ i) : (Basis.mk hli hsp).coord i (v j) = 0 :=
  show hli.repr ⟨v j, Submodule.subset_span (mem_range_self j)⟩ i = 0 by
    /-
      ι : Type u_1
      R : Type u_3
      M : Type u_5
      v : ι → M
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      i j : ι
      h : Ne j i
      ⊢ Eq ((hli.repr ⟨v j, ⋯⟩) i) 0
    -/
    simp [hli.repr_eq_single j, h]
    /-
      🎉 no goals
    -/


/-- Given a basis, the `i`th element of the dual basis evaluates to the Kronecker delta on the
`j`th element of the basis. -/
theorem mk_coord_apply [DecidableEq ι] {i j : ι} :
    (Basis.mk hli hsp).coord i (v j) = if j = i then 1 else 0 := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    v : ι → M
    inst✝³ : Ring R
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    hli : LinearIndependent R v
    hsp : LE.le Top.top (Submodule.span R (Set.range v))
    inst✝ : DecidableEq ι
    i j : ι
    ⊢ Eq (((Basis.mk hli hsp).coord i) (v j)) (ite (Eq j i) 1 0)
  -/
  rcases eq_or_ne j i with h | h
    /-
      case inl
      ι : Type u_1
      R : Type u_3
      M : Type u_5
      v : ι → M
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      inst✝ : DecidableEq ι
      i j : ι
      h : Eq j i
      ⊢ Eq (((Basis.mk hli hsp).coord i) (v j)) (ite (Eq j i) 1 0)
    -/
  · simp only [h, if_true, eq_self_iff_true, mk_coord_apply_eq i]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      R : Type u_3
      M : Type u_5
      v : ι → M
      inst✝³ : Ring R
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      hli : LinearIndependent R v
      hsp : LE.le Top.top (Submodule.span R (Set.range v))
      inst✝ : DecidableEq ι
      i j : ι
      h : Ne j i
      ⊢ Eq (((Basis.mk hli hsp).coord i) (v j)) (ite (Eq j i) 1 0)
    -/
  · simp only [h, if_false, mk_coord_apply_ne h]
    /-
      🎉 no goals
    -/


/-- A linear independent family of vectors is a basis for their span. -/
protected noncomputable def span : Basis ι R (span R (range v)) :=
  Basis.mk (linearIndependent_span hli) <| by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module R₂ M
      x y : M
      b : Basis ι R M
      hli : LinearIndependent R v
      ⊢ LE.le Top.top (Submodule.span R (Set.range fun i => ⟨v i, ⋯⟩))
    -/
    intro x _
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module R₂ M
      x✝ y : M
      b : Basis ι R M
      hli : LinearIndependent R v
      x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
      a✝ : Membership.mem Top.top x
      ⊢ Membership.mem (Submodule.span R (Set.range fun i => ⟨v i, ⋯⟩)) x
    -/
    have : ∀ i, v i ∈ span R (range v) := fun i ↦ subset_span (Set.mem_range_self _)
    have h₁ : (((↑) : span R (range v) → M) '' range fun i => ⟨v i, this i⟩) = range v := by
      simp only [SetLike.coe_sort_coe, ← Set.range_comp]
      rfl
    have h₂ : map (Submodule.subtype (span R (range v))) (span R (range fun i => ⟨v i, this i⟩)) =
        span R (range v) := by
      rw [← span_image, Submodule.coe_subtype]
      -- Porting note: why doesn't `rw [h₁]` work here?
      exact congr_arg _ h₁
    have h₃ : (x : M) ∈ map (Submodule.subtype (span R (range v)))
        (span R (Set.range fun i => Subtype.mk (v i) (this i))) := by
      rw [h₂]
      apply Subtype.mem x
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module R₂ M
      x✝ y : M
      b : Basis ι R M
      hli : LinearIndependent R v
      x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
      a✝ : Membership.mem Top.top x
      this : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range v)) (v i)
      h₁ : Eq (Set.image Subtype.val (Set.range fun i => ⟨v i, ⋯⟩)) (Set.range v)
      h₂ : Eq (Submodule.map (Submodule.span R (Set.range v)).subtype (Submodule.spa …
      h₃ : Membership.mem (Submodule.map (Submodule.span R (Set.range v)).subtype (S …
      ⊢ Membership.mem (Submodule.span R (Set.range fun i => ⟨v i, ⋯⟩)) x
    -/
    rcases mem_map.1 h₃ with ⟨y, hy₁, hy₂⟩
    have h_x_eq_y : x = y := by
      rw [Subtype.ext_iff, ← hy₂]
      simp
    /-
      case intro.intro
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module R₂ M
      x✝ y✝ : M
      b : Basis ι R M
      hli : LinearIndependent R v
      x : Subtype fun x => Membership.mem (Submodule.span R (Set.range v)) x
      a✝ : Membership.mem Top.top x
      this : ∀ (i : ι), Membership.mem (Submodule.span R (Set.range v)) (v i)
      h₁ : Eq (Set.image Subtype.val (Set.range fun i => ⟨v i, ⋯⟩)) (Set.range v)
      h₂ : Eq (Submodule.map (Submodule.span R (Set.range v)).subtype (Submodule.spa …
      h₃ : Membership.mem (Submodule.map (Submodule.span R (Set.range v)).subtype (S …
      y : Subtype (Membership.mem (Submodule.span R (Set.range v)))
      hy₁ : Membership.mem (Submodule.span R (Set.range fun i => ⟨v i, ⋯⟩)) y
      hy₂ : Eq ((Submodule.span R (Set.range v)).subtype y) ↑x
      h_x_eq_y : Eq x y
      ⊢ Membership.mem (Submodule.span R (Set.range fun i => ⟨v i, ⋯⟩)) x
    -/
    rwa [h_x_eq_y]
    /-
      🎉 no goals
    -/


protected theorem span_apply (i : ι) : (Basis.span hli i : M) = v i :=
  congr_arg ((↑) : span R (range v) → M) <| Basis.mk_apply _ _ _


theorem groupSMul_span_eq_top {G : Type*} [Group G] [DistribMulAction G R] [DistribMulAction G M]
    [IsScalarTower G R M] {v : ι → M} (hv : Submodule.span R (Set.range v) = ⊤) {w : ι → G} :
    Submodule.span R (Set.range (w • v)) = ⊤ := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    ⊢ Eq (Submodule.span R (Set.range (HSMul.hSMul w v))) Top.top
  -/
  rw [eq_top_iff]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    ⊢ LE.le Top.top (Submodule.span R (Set.range (HSMul.hSMul w v)))
  -/
  intro j hj
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    j : M
    hj : Membership.mem Top.top j
    ⊢ Membership.mem (Submodule.span R (Set.range (HSMul.hSMul w v))) j
  -/
  rw [← hv] at hj
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    j : M
    hj : Membership.mem (Submodule.span R (Set.range v)) j
    ⊢ Membership.mem (Submodule.span R (Set.range (HSMul.hSMul w v))) j
  -/
  rw [Submodule.mem_span] at hj ⊢
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    j : M
    hj : ∀ (p : Submodule R M), HasSubset.Subset (Set.range v) ↑p → Membership.mem …
    ⊢ ∀ (p : Submodule R M), HasSubset.Subset (Set.range (HSMul.hSMul w v)) ↑p → M …
  -/
  refine fun p hp => hj p fun u hu => ?_
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    j : M
    hj : ∀ (p : Submodule R M), HasSubset.Subset (Set.range v) ↑p → Membership.mem …
    p : Submodule R M
    hp : HasSubset.Subset (Set.range (HSMul.hSMul w v)) ↑p
    u : M
    hu : Membership.mem (Set.range v) u
    ⊢ Membership.mem (↑p) u
  -/
  obtain ⟨i, rfl⟩ := hu
  /-
    case intro
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    j : M
    hj : ∀ (p : Submodule R M), HasSubset.Subset (Set.range v) ↑p → Membership.mem …
    p : Submodule R M
    hp : HasSubset.Subset (Set.range (HSMul.hSMul w v)) ↑p
    i : ι
    ⊢ Membership.mem (↑p) (v i)
  -/
  have : ((w i)⁻¹ • (1 : R)) • w i • v i ∈ p := p.smul_mem ((w i)⁻¹ • (1 : R)) (hp ⟨i, rfl⟩)
  /-
    case intro
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    inst✝⁶ : Ring R
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    G : Type u_7
    inst✝³ : Group G
    inst✝² : DistribMulAction G R
    inst✝¹ : DistribMulAction G M
    inst✝ : IsScalarTower G R M
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    w : ι → G
    j : M
    hj : ∀ (p : Submodule R M), HasSubset.Subset (Set.range v) ↑p → Membership.mem …
    p : Submodule R M
    hp : HasSubset.Subset (Set.range (HSMul.hSMul w v)) ↑p
    i : ι
    this : Membership.mem p (HSMul.hSMul (HSMul.hSMul (Inv.inv (w i)) 1) (HSMul.hS …
    ⊢ Membership.mem (↑p) (v i)
  -/
  rwa [smul_one_smul, inv_smul_smul] at this
  /-
    🎉 no goals
  -/


/-- Given a basis `v` and a map `w` such that for all `i`, `w i` are elements of a group,
`groupSMul` provides the basis corresponding to `w • v`. -/
def groupSMul {G : Type*} [Group G] [DistribMulAction G R] [DistribMulAction G M]
    [IsScalarTower G R M] [SMulCommClass G R M] (v : Basis ι R M) (w : ι → G) : Basis ι R M :=
  Basis.mk (LinearIndependent.group_smul v.linearIndependent w) (groupSMul_span_eq_top v.span_eq).ge


theorem groupSMul_apply {G : Type*} [Group G] [DistribMulAction G R] [DistribMulAction G M]
    [IsScalarTower G R M] [SMulCommClass G R M] {v : Basis ι R M} {w : ι → G} (i : ι) :
    v.groupSMul w i = (w • (v : ι → M)) i :=
  mk_apply (LinearIndependent.group_smul v.linearIndependent w)
    (groupSMul_span_eq_top v.span_eq).ge i


theorem units_smul_span_eq_top {v : ι → M} (hv : Submodule.span R (Set.range v) = ⊤) {w : ι → Rˣ} :
    Submodule.span R (Set.range (w • v)) = ⊤ :=
  groupSMul_span_eq_top hv


/-- Given a basis `v` and a map `w` such that for all `i`, `w i` is a unit, `unitsSMul`
provides the basis corresponding to `w • v`. -/
def unitsSMul (v : Basis ι R M) (w : ι → Rˣ) : Basis ι R M :=
  Basis.mk (LinearIndependent.units_smul v.linearIndependent w)
    (units_smul_span_eq_top v.span_eq).ge


theorem unitsSMul_apply {v : Basis ι R M} {w : ι → Rˣ} (i : ι) : unitsSMul v w i = w i • v i :=
  mk_apply (LinearIndependent.units_smul v.linearIndependent w)
    (units_smul_span_eq_top v.span_eq).ge i


@[simp]
theorem coord_unitsSMul (e : Basis ι R₂ M) (w : ι → R₂ˣ) (i : ι) :
    (unitsSMul e w).coord i = (w i)⁻¹ • e.coord i := by
  classical
    apply e.ext
    intro j
    trans ((unitsSMul e w).coord i) ((w j)⁻¹ • (unitsSMul e w) j)
    · congr
      simp [Basis.unitsSMul, ← mul_smul]
    simp only [Basis.coord_apply, LinearMap.smul_apply, Basis.repr_self, Units.smul_def,
      map_smul, Finsupp.single_apply]
    split_ifs with h <;> simp [h]


@[simp]
theorem repr_unitsSMul (e : Basis ι R₂ M) (w : ι → R₂ˣ) (v : M) (i : ι) :
    (e.unitsSMul w).repr v i = (w i)⁻¹ • e.repr v i :=
  congr_arg (fun f : M →ₗ[R₂] R₂ => f v) (e.coord_unitsSMul w i)


/-- A version of `unitsSMul` that uses `IsUnit`. -/
def isUnitSMul (v : Basis ι R M) {w : ι → R} (hw : ∀ i, IsUnit (w i)) : Basis ι R M :=
  unitsSMul v fun i => (hw i).unit


theorem isUnitSMul_apply {v : Basis ι R M} {w : ι → R} (hw : ∀ i, IsUnit (w i)) (i : ι) :
    v.isUnitSMul hw i = w i • v i :=
  unitsSMul_apply i


theorem repr_isUnitSMul {v : Basis ι R₂ M} {w : ι → R₂} (hw : ∀ i, IsUnit (w i)) (x : M) (i : ι) :
    (v.isUnitSMul hw).repr x i = (hw i).unit⁻¹ • v.repr x i :=
  repr_unitsSMul _ _ _ _


/-- Let `b` be a basis for a submodule `N` of `M`. If `y : M` is linear independent of `N`
and `y` and `N` together span the whole of `M`, then there is a basis for `M`
whose basis vectors are given by `Fin.cons y b`. -/
noncomputable def mkFinCons {n : ℕ} {N : Submodule R M} (y : M) (b : Basis (Fin n) R N)
    (hli : ∀ (c : R), ∀ x ∈ N, c • y + x = 0 → c = 0) (hsp : ∀ z : M, ∃ c : R, z + c • y ∈ N) :
    Basis (Fin (n + 1)) R M :=
  have span_b : Submodule.span R (Set.range (N.subtype ∘ b)) = N := by
    /-
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      v : ι → M
      inst✝⁴ : Ring R
      inst✝³ : CommRing R₂
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : Module R₂ M
      x y✝ : M
      b✝ : Basis ι R M
      n : Nat
      N : Submodule R M
      y : M
      b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
      hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
      hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
      ⊢ Eq (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b))) N
    -/
    rw [Set.range_comp, Submodule.span_image, b.span_eq, Submodule.map_subtype_top]
    /-
      🎉 no goals
    -/
  Basis.mk (v := Fin.cons y (N.subtype ∘ b))
    ((b.linearIndependent.map' N.subtype (Submodule.ker_subtype _)).fin_cons' _ _
      (by
        /-
          ι : Type u_1
          ι' : Type u_2
          R : Type u_3
          R₂ : Type u_4
          M : Type u_5
          M' : Type u_6
          v : ι → M
          inst✝⁴ : Ring R
          inst✝³ : CommRing R₂
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module R₂ M
          x y✝ : M
          b✝ : Basis ι R M
          n : Nat
          N : Submodule R M
          y : M
          b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
          hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
          hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
          span_b : Eq (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b))) N
          ⊢ ∀ (c : R) (y_1 : Subtype fun x => Membership.mem (Submodule.span R (Set.rang …
        -/
        rintro c ⟨x, hx⟩ hc
        /-
          case mk
          ι : Type u_1
          ι' : Type u_2
          R : Type u_3
          R₂ : Type u_4
          M : Type u_5
          M' : Type u_6
          v : ι → M
          inst✝⁴ : Ring R
          inst✝³ : CommRing R₂
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module R₂ M
          x✝ y✝ : M
          b✝ : Basis ι R M
          n : Nat
          N : Submodule R M
          y : M
          b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
          hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
          hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
          span_b : Eq (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b))) N
          c : R
          x : M
          hx : Membership.mem (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b) …
          hc : Eq (HAdd.hAdd (HSMul.hSMul c y) ↑⟨x, hx⟩) 0
          ⊢ Eq c 0
        -/
        rw [span_b] at hx
        /-
          case mk
          ι : Type u_1
          ι' : Type u_2
          R : Type u_3
          R₂ : Type u_4
          M : Type u_5
          M' : Type u_6
          v : ι → M
          inst✝⁴ : Ring R
          inst✝³ : CommRing R₂
          inst✝² : AddCommGroup M
          inst✝¹ : Module R M
          inst✝ : Module R₂ M
          x✝ y✝ : M
          b✝ : Basis ι R M
          n : Nat
          N : Submodule R M
          y : M
          b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
          hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
          hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
          span_b : Eq (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b))) N
          c : R
          x : M
          hx✝ : Membership.mem (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b …
          hx : Membership.mem N x
          hc : Eq (HAdd.hAdd (HSMul.hSMul c y) ↑⟨x, hx✝⟩) 0
          ⊢ Eq c 0
        -/
        exact hli c x hx hc))
        /-
          🎉 no goals
        -/
    fun x _ => by
      /-
        ι : Type u_1
        ι' : Type u_2
        R : Type u_3
        R₂ : Type u_4
        M : Type u_5
        M' : Type u_6
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : CommRing R₂
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : Module R₂ M
        x✝¹ y✝ : M
        b✝ : Basis ι R M
        n : Nat
        N : Submodule R M
        y : M
        b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
        hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
        hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
        span_b : Eq (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b))) N
        x : M
        x✝ : Membership.mem Top.top x
        ⊢ Membership.mem (Submodule.span R (Set.range (Fin.cons y (Function.comp ⇑N.su …
      -/
      rw [Fin.range_cons, Submodule.mem_span_insert', span_b]
      /-
        ι : Type u_1
        ι' : Type u_2
        R : Type u_3
        R₂ : Type u_4
        M : Type u_5
        M' : Type u_6
        v : ι → M
        inst✝⁴ : Ring R
        inst✝³ : CommRing R₂
        inst✝² : AddCommGroup M
        inst✝¹ : Module R M
        inst✝ : Module R₂ M
        x✝¹ y✝ : M
        b✝ : Basis ι R M
        n : Nat
        N : Submodule R M
        y : M
        b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
        hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
        hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
        span_b : Eq (Submodule.span R (Set.range (Function.comp ⇑N.subtype ⇑b))) N
        x : M
        x✝ : Membership.mem Top.top x
        ⊢ Exists fun a => Membership.mem N (HAdd.hAdd x (HSMul.hSMul a y))
      -/
      exact hsp x
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_mkFinCons {n : ℕ} {N : Submodule R M} (y : M) (b : Basis (Fin n) R N)
    (hli : ∀ (c : R), ∀ x ∈ N, c • y + x = 0 → c = 0) (hsp : ∀ z : M, ∃ c : R, z + c • y ∈ N) :
    (mkFinCons y b hli hsp : Fin (n + 1) → M) = Fin.cons y ((↑) ∘ b) := by
  -- Porting note: without `unfold`, Lean can't reuse the proofs included in the definition
  -- `mkFinCons`
  /-
    R : Type u_3
    M : Type u_5
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    y : M
    b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
    hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
    hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
    ⊢ Eq (⇑(Basis.mkFinCons y b hli hsp)) (Fin.cons y (Function.comp Subtype.val ⇑ …
  -/
  unfold mkFinCons
  /-
    R : Type u_3
    M : Type u_5
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    n : Nat
    N : Submodule R M
    y : M
    b : Basis (Fin n) R (Subtype fun x => Membership.mem N x)
    hli : ∀ (c : R) (x : M), Membership.mem N x → Eq (HAdd.hAdd (HSMul.hSMul c y)  …
    hsp : ∀ (z : M), Exists fun c => Membership.mem N (HAdd.hAdd z (HSMul.hSMul c  …
    ⊢ Eq (⇑(letFun ⋯ fun span_b => Basis.mk ⋯ ⋯)) (Fin.cons y (Function.comp Subty …
  -/
  exact coe_mk (v := Fin.cons y (N.subtype ∘ b)) _ _
  /-
    🎉 no goals
  -/


/-- Let `b` be a basis for a submodule `N ≤ O`. If `y ∈ O` is linear independent of `N`
and `y` and `N` together span the whole of `O`, then there is a basis for `O`
whose basis vectors are given by `Fin.cons y b`. -/
noncomputable def mkFinConsOfLE {n : ℕ} {N O : Submodule R M} (y : M) (yO : y ∈ O)
    (b : Basis (Fin n) R N) (hNO : N ≤ O) (hli : ∀ (c : R), ∀ x ∈ N, c • y + x = 0 → c = 0)
    (hsp : ∀ z ∈ O, ∃ c : R, z + c • y ∈ N) : Basis (Fin (n + 1)) R O :=
  mkFinCons ⟨y, yO⟩ (b.map (Submodule.comapSubtypeEquivOfLe hNO).symm)
    (fun c x hc hx => hli c x (Submodule.mem_comap.mp hc) (congr_arg ((↑) : O → M) hx))
    fun z => hsp z z.2


@[simp]
theorem coe_mkFinConsOfLE {n : ℕ} {N O : Submodule R M} (y : M) (yO : y ∈ O) (b : Basis (Fin n) R N)
    (hNO : N ≤ O) (hli : ∀ (c : R), ∀ x ∈ N, c • y + x = 0 → c = 0)
    (hsp : ∀ z ∈ O, ∃ c : R, z + c • y ∈ N) :
    (mkFinConsOfLE y yO b hNO hli hsp : Fin (n + 1) → O) =
      Fin.cons ⟨y, yO⟩ (Submodule.inclusion hNO ∘ b) :=
  coe_mkFinCons _ _ _ _


/-- The basis of `R × R` given by the two vectors `(1, 0)` and `(0, 1)`. -/
protected def finTwoProd (R : Type*) [Semiring R] : Basis (Fin 2) R (R × R) :=
  Basis.ofEquivFun (LinearEquiv.finTwoArrow R R).symm


@[simp]
theorem finTwoProd_zero (R : Type*) [Semiring R] : Basis.finTwoProd R 0 = (1, 0) := by
  /-
    R : Type u_7
    inst✝ : Semiring R
    ⊢ Eq ((Basis.finTwoProd R) 0) { fst := 1, snd := 0 }
  -/
  simp [Basis.finTwoProd, LinearEquiv.finTwoArrow]
  /-
    🎉 no goals
  -/


@[simp]
theorem finTwoProd_one (R : Type*) [Semiring R] : Basis.finTwoProd R 1 = (0, 1) := by
  /-
    R : Type u_7
    inst✝ : Semiring R
    ⊢ Eq ((Basis.finTwoProd R) 1) { fst := 0, snd := 1 }
  -/
  simp [Basis.finTwoProd, LinearEquiv.finTwoArrow]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_finTwoProd_repr {R : Type*} [Semiring R] (x : R × R) :
    ⇑((Basis.finTwoProd R).repr x) = ![x.fst, x.snd] :=
  rfl


/-- If `N` is a submodule with finite rank, do induction on adjoining a linear independent
element to a submodule. -/
def Submodule.inductionOnRankAux (b : Basis ι R M) (P : Submodule R M → Sort*)
    (ih : ∀ N : Submodule R M,
      (∀ N' ≤ N, ∀ x ∈ N, (∀ (c : R), ∀ y ∈ N', c • x + y = (0 : M) → c = 0) → P N') → P N)
    (n : ℕ) (N : Submodule R M)
    (rank_le : ∀ {m : ℕ} (v : Fin m → N), LinearIndependent R ((↑) ∘ v : Fin m → M) → m ≤ n) :
    P N := by
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    n : Nat
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    ⊢ P N
  -/
  haveI : DecidableEq M := Classical.decEq M
  have Pbot : P ⊥ := by
    apply ih
    intro N _ x x_mem x_ortho
    exfalso
    rw [mem_bot] at x_mem
    simpa [x_mem] using x_ortho 1 0 N.zero_mem
  /-
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    n : Nat
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    this : DecidableEq M
    Pbot : P Bot.bot
    ⊢ P N
  -/
  induction' n with n rank_ih generalizing N
    /-
      case zero
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      ⊢ P N
    -/
  · suffices N = ⊥ by rwa [this]
    /-
      case zero
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      ⊢ Eq N Bot.bot
    -/
    apply Basis.eq_bot_of_rank_eq_zero b _ fun m hv => Nat.le_zero.mp (rank_le _ hv)
    /-
      🎉 no goals
    -/
  /-
    case succ
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    this : DecidableEq M
    Pbot : P Bot.bot
    n : Nat
    rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    ⊢ P N
  -/
  apply ih
  /-
    case succ.a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    this : DecidableEq M
    Pbot : P Bot.bot
    n : Nat
    rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    ⊢ (N' : Submodule R M) → LE.le N' N → (x : M) → Membership.mem N x → (∀ (c : R …
  -/
  intro N' N'_le x x_mem x_ortho
  /-
    case succ.a
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    this : DecidableEq M
    Pbot : P Bot.bot
    n : Nat
    rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    N' : Submodule R M
    N'_le : LE.le N' N
    x : M
    x_mem : Membership.mem N x
    x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
    ⊢ P N'
  -/
  apply rank_ih
  /-
    case succ.a.rank_le
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    this : DecidableEq M
    Pbot : P Bot.bot
    n : Nat
    rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    N' : Submodule R M
    N'_le : LE.le N' N
    x : M
    x_mem : Membership.mem N x
    x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
    ⊢ ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N' x), LinearIndepe …
  -/
  intro m v hli
  /-
    case succ.a.rank_le
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    this : DecidableEq M
    Pbot : P Bot.bot
    n : Nat
    rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    N' : Submodule R M
    N'_le : LE.le N' N
    x : M
    x_mem : Membership.mem N x
    x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
    m : Nat
    v : Fin m → Subtype fun x => Membership.mem N' x
    hli : LinearIndependent R (Function.comp Subtype.val v)
    ⊢ LE.le m n
  -/
  refine Nat.succ_le_succ_iff.mp (rank_le (Fin.cons ⟨x, x_mem⟩ fun i => ⟨v i, N'_le (v i).2⟩) ?_)
  /-
    case succ.a.rank_le
    ι : Type u_1
    ι' : Type u_2
    R : Type u_3
    R₂ : Type u_4
    M : Type u_5
    M' : Type u_6
    inst✝³ : Ring R
    inst✝² : IsDomain R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    b✝ : ι → M
    b : Basis ι R M
    P : Submodule R M → Sort u_7
    ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
    this : DecidableEq M
    Pbot : P Bot.bot
    n : Nat
    rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
    N : Submodule R M
    rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
    N' : Submodule R M
    N'_le : LE.le N' N
    x : M
    x_mem : Membership.mem N x
    x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
    m : Nat
    v : Fin m → Subtype fun x => Membership.mem N' x
    hli : LinearIndependent R (Function.comp Subtype.val v)
    ⊢ LinearIndependent R (Function.comp Subtype.val (Fin.cons ⟨x, x_mem⟩ fun i => …
  -/
  convert hli.fin_cons' x _ ?_
    /-
      case h.e'_4
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      n : Nat
      rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      N' : Submodule R M
      N'_le : LE.le N' N
      x : M
      x_mem : Membership.mem N x
      x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
      m : Nat
      v : Fin m → Subtype fun x => Membership.mem N' x
      hli : LinearIndependent R (Function.comp Subtype.val v)
      ⊢ Eq (Function.comp Subtype.val (Fin.cons ⟨x, x_mem⟩ fun i => ⟨↑(v i), ⋯⟩)) (F …
    -/
  · ext i
    /-
      case h.e'_4.h
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      n : Nat
      rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      N' : Submodule R M
      N'_le : LE.le N' N
      x : M
      x_mem : Membership.mem N x
      x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
      m : Nat
      v : Fin m → Subtype fun x => Membership.mem N' x
      hli : LinearIndependent R (Function.comp Subtype.val v)
      i : Fin m.succ
      ⊢ Eq (Function.comp Subtype.val (Fin.cons ⟨x, x_mem⟩ fun i => ⟨↑(v i), ⋯⟩) i)  …
    -/
                                 /-
                                   🎉 no goals
                                 -/
    refine Fin.cases ?_ ?_ i <;> simp
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case succ.a.rank_le
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      n : Nat
      rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      N' : Submodule R M
      N'_le : LE.le N' N
      x : M
      x_mem : Membership.mem N x
      x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
      m : Nat
      v : Fin m → Subtype fun x => Membership.mem N' x
      hli : LinearIndependent R (Function.comp Subtype.val v)
      ⊢ ∀ (c : R) (y : Subtype fun x => Membership.mem (Submodule.span R (Set.range  …
    -/
  · intro c y hcy
    /-
      case succ.a.rank_le
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      n : Nat
      rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      N' : Submodule R M
      N'_le : LE.le N' N
      x : M
      x_mem : Membership.mem N x
      x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
      m : Nat
      v : Fin m → Subtype fun x => Membership.mem N' x
      hli : LinearIndependent R (Function.comp Subtype.val v)
      c : R
      y : Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function.com …
      hcy : Eq (HAdd.hAdd (HSMul.hSMul c x) ↑y) 0
      ⊢ Eq c 0
    -/
    refine x_ortho c y (Submodule.span_le.mpr ?_ y.2) hcy
    /-
      case succ.a.rank_le
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      n : Nat
      rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      N' : Submodule R M
      N'_le : LE.le N' N
      x : M
      x_mem : Membership.mem N x
      x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
      m : Nat
      v : Fin m → Subtype fun x => Membership.mem N' x
      hli : LinearIndependent R (Function.comp Subtype.val v)
      c : R
      y : Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function.com …
      hcy : Eq (HAdd.hAdd (HSMul.hSMul c x) ↑y) 0
      ⊢ HasSubset.Subset (Set.range (Function.comp Subtype.val v)) ↑N'
    -/
    rintro _ ⟨z, rfl⟩
    /-
      case succ.a.rank_le.intro
      ι : Type u_1
      ι' : Type u_2
      R : Type u_3
      R₂ : Type u_4
      M : Type u_5
      M' : Type u_6
      inst✝³ : Ring R
      inst✝² : IsDomain R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      b✝ : ι → M
      b : Basis ι R M
      P : Submodule R M → Sort u_7
      ih : (N : Submodule R M) → ((N' : Submodule R M) → LE.le N' N → (x : M) → Memb …
      this : DecidableEq M
      Pbot : P Bot.bot
      n : Nat
      rank_ih : (N : Submodule R M) → (∀ {m : Nat} (v : Fin m → Subtype fun x => Mem …
      N : Submodule R M
      rank_le : ∀ {m : Nat} (v : Fin m → Subtype fun x => Membership.mem N x), Linea …
      N' : Submodule R M
      N'_le : LE.le N' N
      x : M
      x_mem : Membership.mem N x
      x_ortho : ∀ (c : R) (y : M), Membership.mem N' y → Eq (HAdd.hAdd (HSMul.hSMul  …
      m : Nat
      v : Fin m → Subtype fun x => Membership.mem N' x
      hli : LinearIndependent R (Function.comp Subtype.val v)
      c : R
      y : Subtype fun x => Membership.mem (Submodule.span R (Set.range (Function.com …
      hcy : Eq (HAdd.hAdd (HSMul.hSMul c x) ↑y) 0
      z : Fin m
      ⊢ Membership.mem (↑N') (Function.comp Subtype.val v z)
    -/
    exact (v z).2
    /-
      🎉 no goals
    -/


/-- An element of a non-unital-non-associative algebra is in the center exactly when it commutes
with the basis elements. -/
lemma Basis.mem_center_iff {A}
    [Semiring R] [NonUnitalNonAssocSemiring A]
    [Module R A] [SMulCommClass R A A] [SMulCommClass R R A] [IsScalarTower R A A]
    (b : Basis ι R A) {z : A} :
    z ∈ Set.center A ↔
      (∀ i, Commute (b i) z) ∧ ∀ i j,
        z * (b i * b j) = (z * b i) * b j
          ∧ (b i * z) * b j = b i * (z * b j)
          ∧ (b i * b j) * z = b i * (b j * z) := by
  /-
    ι : Type u_1
    R : Type u_3
    A : Type u_7
    inst✝⁵ : Semiring R
    inst✝⁴ : NonUnitalNonAssocSemiring A
    inst✝³ : Module R A
    inst✝² : SMulCommClass R A A
    inst✝¹ : SMulCommClass R R A
    inst✝ : IsScalarTower R A A
    b : Basis ι R A
    z : A
    ⊢ Iff (Membership.mem (Set.center A) z) (And (∀ (i : ι), Commute (b i) z) (∀ ( …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      R : Type u_3
      A : Type u_7
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring A
      inst✝³ : Module R A
      inst✝² : SMulCommClass R A A
      inst✝¹ : SMulCommClass R R A
      inst✝ : IsScalarTower R A A
      b : Basis ι R A
      z : A
      ⊢ Membership.mem (Set.center A) z → And (∀ (i : ι), Commute (b i) z) (∀ (i j : …
    -/
  · intro h
    /-
      case mp
      ι : Type u_1
      R : Type u_3
      A : Type u_7
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring A
      inst✝³ : Module R A
      inst✝² : SMulCommClass R A A
      inst✝¹ : SMulCommClass R R A
      inst✝ : IsScalarTower R A A
      b : Basis ι R A
      z : A
      h : Membership.mem (Set.center A) z
      ⊢ And (∀ (i : ι), Commute (b i) z) (∀ (i j : ι), And (Eq (HMul.hMul z (HMul.hM …
    -/
    constructor
      /-
        case mp.left
        ι : Type u_1
        R : Type u_3
        A : Type u_7
        inst✝⁵ : Semiring R
        inst✝⁴ : NonUnitalNonAssocSemiring A
        inst✝³ : Module R A
        inst✝² : SMulCommClass R A A
        inst✝¹ : SMulCommClass R R A
        inst✝ : IsScalarTower R A A
        b : Basis ι R A
        z : A
        h : Membership.mem (Set.center A) z
        ⊢ ∀ (i : ι), Commute (b i) z
      -/
    · intro i
      /-
        case mp.left
        ι : Type u_1
        R : Type u_3
        A : Type u_7
        inst✝⁵ : Semiring R
        inst✝⁴ : NonUnitalNonAssocSemiring A
        inst✝³ : Module R A
        inst✝² : SMulCommClass R A A
        inst✝¹ : SMulCommClass R R A
        inst✝ : IsScalarTower R A A
        b : Basis ι R A
        z : A
        h : Membership.mem (Set.center A) z
        i : ι
        ⊢ Commute (b i) z
      -/
      apply (h.1 (b i)).symm
      /-
        🎉 no goals
      -/
      /-
        case mp.right
        ι : Type u_1
        R : Type u_3
        A : Type u_7
        inst✝⁵ : Semiring R
        inst✝⁴ : NonUnitalNonAssocSemiring A
        inst✝³ : Module R A
        inst✝² : SMulCommClass R A A
        inst✝¹ : SMulCommClass R R A
        inst✝ : IsScalarTower R A A
        b : Basis ι R A
        z : A
        h : Membership.mem (Set.center A) z
        ⊢ ∀ (i j : ι), And (Eq (HMul.hMul z (HMul.hMul (b i) (b j))) (HMul.hMul (HMul. …
      -/
    · intros
      /-
        case mp.right
        ι : Type u_1
        R : Type u_3
        A : Type u_7
        inst✝⁵ : Semiring R
        inst✝⁴ : NonUnitalNonAssocSemiring A
        inst✝³ : Module R A
        inst✝² : SMulCommClass R A A
        inst✝¹ : SMulCommClass R R A
        inst✝ : IsScalarTower R A A
        b : Basis ι R A
        z : A
        h : Membership.mem (Set.center A) z
        i✝ j✝ : ι
        ⊢ And (Eq (HMul.hMul z (HMul.hMul (b i✝) (b j✝))) (HMul.hMul (HMul.hMul z (b i …
      -/
      exact ⟨h.2 _ _, ⟨h.3 _ _, h.4 _ _⟩⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      ι : Type u_1
      R : Type u_3
      A : Type u_7
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring A
      inst✝³ : Module R A
      inst✝² : SMulCommClass R A A
      inst✝¹ : SMulCommClass R R A
      inst✝ : IsScalarTower R A A
      b : Basis ι R A
      z : A
      ⊢ And (∀ (i : ι), Commute (b i) z) (∀ (i j : ι), And (Eq (HMul.hMul z (HMul.hM …
    -/
  · intro h
    /-
      case mpr
      ι : Type u_1
      R : Type u_3
      A : Type u_7
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring A
      inst✝³ : Module R A
      inst✝² : SMulCommClass R A A
      inst✝¹ : SMulCommClass R R A
      inst✝ : IsScalarTower R A A
      b : Basis ι R A
      z : A
      h : And (∀ (i : ι), Commute (b i) z) (∀ (i j : ι), And (Eq (HMul.hMul z (HMul. …
      ⊢ Membership.mem (Set.center A) z
    -/
    rw [center, mem_setOf_eq]
    /-
      case mpr
      ι : Type u_1
      R : Type u_3
      A : Type u_7
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring A
      inst✝³ : Module R A
      inst✝² : SMulCommClass R A A
      inst✝¹ : SMulCommClass R R A
      inst✝ : IsScalarTower R A A
      b : Basis ι R A
      z : A
      h : And (∀ (i : ι), Commute (b i) z) (∀ (i j : ι), And (Eq (HMul.hMul z (HMul. …
      ⊢ IsMulCentral z
    -/
    constructor
    case comm =>
      intro y
      rw [← b.linearCombination_repr y, linearCombination_apply, sum, Finset.sum_mul,
          Finset.mul_sum]
      simp_rw [mul_smul_comm, smul_mul_assoc, (h.1 _).eq]
    case left_assoc =>
      intro c d
      rw [← b.linearCombination_repr c, ← b.linearCombination_repr d, linearCombination_apply,
          linearCombination_apply, sum, sum, Finset.sum_mul, Finset.mul_sum, Finset.mul_sum,
          Finset.mul_sum]
      simp_rw [smul_mul_assoc, Finset.mul_sum, Finset.sum_mul, mul_smul_comm, Finset.mul_sum,
        Finset.smul_sum, smul_mul_assoc, mul_smul_comm, (h.2 _ _).1,
        (@SMulCommClass.smul_comm R R A)]
      rw [Finset.sum_comm]
    case mid_assoc =>
      intro c d
      rw [← b.linearCombination_repr c, ← b.linearCombination_repr d, linearCombination_apply,
          linearCombination_apply, sum, sum, Finset.sum_mul, Finset.mul_sum, Finset.mul_sum,
          Finset.mul_sum]
      simp_rw [smul_mul_assoc, Finset.sum_mul, mul_smul_comm, smul_mul_assoc, (h.2 _ _).2.1]
    case right_assoc =>
      intro c d
      rw [← b.linearCombination_repr c, ← b.linearCombination_repr d, linearCombination_apply,
          linearCombination_apply, sum, Finsupp.sum, Finset.sum_mul]
      simp_rw [smul_mul_assoc, Finset.mul_sum, Finset.sum_mul, mul_smul_comm, Finset.mul_sum,
               Finset.smul_sum, smul_mul_assoc, mul_smul_comm, Finset.sum_mul, smul_mul_assoc,
               (h.2 _ _).2.2]


/-- Let `b` be an `S`-basis of `M`. Let `R` be a CommRing such that `Algebra R S` has no zero smul
divisors, then the submodule of `M` spanned by `b` over `R` admits `b` as an `R`-basis. -/
noncomputable def Basis.restrictScalars : Basis ι R (span R (Set.range b)) :=
  Basis.span (b.linearIndependent.restrict_scalars (smul_left_injective R one_ne_zero))


@[simp]
theorem Basis.restrictScalars_apply (i : ι) : (b.restrictScalars R i : M) = b i := by
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    i : ι
    ⊢ Eq (↑((Basis.restrictScalars R b) i)) (b i)
  -/
  simp only [Basis.restrictScalars, Basis.span_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem Basis.restrictScalars_repr_apply (m : span R (Set.range b)) (i : ι) :
    algebraMap R S ((b.restrictScalars R).repr m i) = b.repr m i := by
  suffices
    Finsupp.mapRange.linearMap (Algebra.linearMap R S) ∘ₗ (b.restrictScalars R).repr.toLinearMap =
      ((b.repr : M →ₗ[S] ι →₀ S).restrictScalars R).domRestrict _
    by exact DFunLike.congr_fun (LinearMap.congr_fun this m) i
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    m : Subtype fun x => Membership.mem (Submodule.span R (Set.range ⇑b)) x
    i : ι
    ⊢ Eq ((Finsupp.mapRange.linearMap (Algebra.linearMap R S)).comp ↑(Basis.restri …
  -/
  refine Basis.ext (b.restrictScalars R) fun _ => ?_
  simp only [LinearMap.coe_comp, LinearEquiv.coe_toLinearMap, Function.comp_apply, map_one,
    Basis.repr_self, Finsupp.mapRange.linearMap_apply, Finsupp.mapRange_single,
    Algebra.linearMap_apply, LinearMap.domRestrict_apply, LinearEquiv.coe_coe,
    Basis.restrictScalars_apply, LinearMap.coe_restrictScalars]


/-- Let `b` be an `S`-basis of `M`. Then `m : M` lies in the `R`-module spanned by `b` iff all the
coordinates of `m` on the basis `b` are in `R` (see `Basis.mem_span` for the case `R = S`). -/
theorem Basis.mem_span_iff_repr_mem (m : M) :
    m ∈ span R (Set.range b) ↔ ∀ i, b.repr m i ∈ Set.range (algebraMap R S) := by
  refine
    ⟨fun hm i => ⟨(b.restrictScalars R).repr ⟨m, hm⟩ i, b.restrictScalars_repr_apply R ⟨m, hm⟩ i⟩,
      fun h => ?_⟩
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    m : M
    h : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap R S)) ((b.repr m) i)
    ⊢ Membership.mem (Submodule.span R (Set.range ⇑b)) m
  -/
  rw [← b.linearCombination_repr m, Finsupp.linearCombination_apply S _]
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    m : M
    h : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap R S)) ((b.repr m) i)
    ⊢ Membership.mem (Submodule.span R (Set.range ⇑b)) ((b.repr m).sum fun i a =>  …
  -/
  refine sum_mem fun i _ => ?_
  /-
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    m : M
    h : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap R S)) ((b.repr m) i)
    i : ι
    x✝ : Membership.mem (b.repr m).support i
    ⊢ Membership.mem (Submodule.span R (Set.range ⇑b)) ((fun i a => HSMul.hSMul a  …
  -/
  obtain ⟨_, h⟩ := h i
  /-
    case intro
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    m : M
    h✝ : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap R S)) ((b.repr m) i)
    i : ι
    x✝ : Membership.mem (b.repr m).support i
    w✝ : R
    h : Eq ((algebraMap R S) w✝) ((b.repr m) i)
    ⊢ Membership.mem (Submodule.span R (Set.range ⇑b)) ((fun i a => HSMul.hSMul a  …
  -/
  simp_rw [← h, algebraMap_smul]
  /-
    case intro
    ι : Type u_1
    R : Type u_3
    M : Type u_5
    S : Type u_7
    inst✝⁸ : CommRing R
    inst✝⁷ : Ring S
    inst✝⁶ : Nontrivial S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module S M
    inst✝² : Module R M
    inst✝¹ : IsScalarTower R S M
    inst✝ : NoZeroSMulDivisors R S
    b : Basis ι S M
    m : M
    h✝ : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap R S)) ((b.repr m) i)
    i : ι
    x✝ : Membership.mem (b.repr m).support i
    w✝ : R
    h : Eq ((algebraMap R S) w✝) ((b.repr m) i)
    ⊢ Membership.mem (Submodule.span R (Set.range ⇑b)) (HSMul.hSMul w✝ (b i))
  -/
  exact smul_mem _ _ (subset_span (Set.mem_range_self i))
  /-
    🎉 no goals
  -/


