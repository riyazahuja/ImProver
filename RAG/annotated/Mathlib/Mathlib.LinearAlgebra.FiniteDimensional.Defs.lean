/-- `FiniteDimensional` vector spaces are defined to be finite modules.
Use `FiniteDimensional.of_fintype_basis` to prove finite dimension from another definition. -/
abbrev FiniteDimensional (K V : Type*) [DivisionRing K] [AddCommGroup V] [Module K V] :=
  Module.Finite K V


/-- If the codomain of an injective linear map is finite dimensional, the domain must be as well. -/
theorem of_injective (f : V →ₗ[K] V₂) (w : Function.Injective f) [FiniteDimensional K V₂] :
    FiniteDimensional K V :=
  have : IsNoetherian K V₂ := IsNoetherian.iff_fg.mpr ‹_›
  Module.Finite.of_injective f w


/-- If the domain of a surjective linear map is finite dimensional, the codomain must be as well. -/
theorem of_surjective (f : V →ₗ[K] V₂) (w : Function.Surjective f) [FiniteDimensional K V] :
    FiniteDimensional K V₂ :=
  Module.Finite.of_surjective f w


instance finiteDimensional_pi {ι : Type*} [Finite ι] : FiniteDimensional K (ι → K) :=
  Finite.pi


instance finiteDimensional_pi' {ι : Type*} [Finite ι] (M : ι → Type*) [∀ i, AddCommGroup (M i)]
    [∀ i, Module K (M i)] [∀ i, FiniteDimensional K (M i)] : FiniteDimensional K (∀ i, M i) :=
  Finite.pi


/-- If a vector space has a finite basis, then it is finite-dimensional. -/
theorem of_fintype_basis {ι : Type w} [Finite ι] (h : Basis ι K V) : FiniteDimensional K V :=
  Module.Finite.of_basis h


/-- If a vector space is `FiniteDimensional`, all bases are indexed by a finite type -/
noncomputable def fintypeBasisIndex {ι : Type*} [FiniteDimensional K V] (b : Basis ι K V) :
    Fintype ι :=
  @Fintype.ofFinite _ (Module.Finite.finite_basis b)


/-- If a vector space is `FiniteDimensional`, `Basis.ofVectorSpace` is indexed by
  a finite type. -/
noncomputable instance [FiniteDimensional K V] : Fintype (Basis.ofVectorSpaceIndex K V) := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    V₂ : Type v'
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    inst✝ : FiniteDimensional K V
    ⊢ Fintype ↑(Basis.ofVectorSpaceIndex K V)
  -/
  letI : IsNoetherian K V := IsNoetherian.iff_fg.2 inferInstance
  /-
    K : Type u
    V : Type v
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    V₂ : Type v'
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    inst✝ : FiniteDimensional K V
    this : IsNoetherian K V := IsNoetherian.iff_fg.mpr inferInstance
    ⊢ Fintype ↑(Basis.ofVectorSpaceIndex K V)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If a vector space has a basis indexed by elements of a finite set, then it is
finite-dimensional. -/
theorem of_finite_basis {ι : Type w} {s : Set ι} (h : Basis s K V) (hs : Set.Finite s) :
    FiniteDimensional K V :=
  haveI := hs.fintype
  of_fintype_basis h


/-- A subspace of a finite-dimensional space is also finite-dimensional. -/
instance finiteDimensional_submodule [FiniteDimensional K V] (S : Submodule K V) :
    FiniteDimensional K S := by
  /-
    K : Type u
    V : Type v
    inst✝⁵ : DivisionRing K
    inst✝⁴ : AddCommGroup V
    inst✝³ : Module K V
    V₂ : Type v'
    inst✝² : AddCommGroup V₂
    inst✝¹ : Module K V₂
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem S x)
  -/
  letI : IsNoetherian K V := iff_fg.2 ?_
  · exact iff_fg.1 <| IsNoetherian.iff_rank_lt_aleph0.2 <|
      (Submodule.rank_le _).trans_lt (rank_lt_aleph0 K V)
    /-
      case refine_1
      K : Type u
      V : Type v
      inst✝⁵ : DivisionRing K
      inst✝⁴ : AddCommGroup V
      inst✝³ : Module K V
      V₂ : Type v'
      inst✝² : AddCommGroup V₂
      inst✝¹ : Module K V₂
      inst✝ : FiniteDimensional K V
      S : Submodule K V
      ⊢ Module.Finite K V
    -/
  · infer_instance
    /-
      🎉 no goals
    -/


/-- A quotient of a finite-dimensional space is also finite-dimensional. -/
instance finiteDimensional_quotient [FiniteDimensional K V] (S : Submodule K V) :
    FiniteDimensional K (V ⧸ S) :=
  Module.Finite.quotient K S


theorem of_finrank_pos (h : 0 < finrank K V) : FiniteDimensional K V :=
  Module.finite_of_finrank_pos h


theorem of_finrank_eq_succ {n : ℕ} (hn : finrank K V = n.succ) :
    FiniteDimensional K V :=
  Module.finite_of_finrank_eq_succ hn


/-- We can infer `FiniteDimensional K V` in the presence of `[Fact (finrank K V = n + 1)]`. Declare
this as a local instance where needed. -/
theorem of_fact_finrank_eq_succ (n : ℕ) [hn : Fact (finrank K V = n + 1)] :
    FiniteDimensional K V :=
  of_finrank_eq_succ hn.out


/-- In a finite-dimensional space, its dimension (seen as a cardinal) coincides with its
`finrank`. This is a copy of `finrank_eq_rank _ _` which creates easier typeclass searches. -/
theorem finrank_eq_rank' [FiniteDimensional K V] : (finrank K V : Cardinal.{v}) = Module.rank K V :=
  finrank_eq_rank _ _


theorem finrank_of_infinite_dimensional (h : ¬FiniteDimensional K V) : finrank K V = 0 :=
  Module.finrank_of_not_finite h


theorem finiteDimensional_iff_of_rank_eq_nsmul {W} [AddCommGroup W] [Module K W] {n : ℕ}
    (hn : n ≠ 0) (hVW : Module.rank K V = n • Module.rank K W) :
    FiniteDimensional K V ↔ FiniteDimensional K W :=
  Module.finite_iff_of_rank_eq_nsmul hn hVW


/-- If a vector space is finite-dimensional, then the cardinality of any basis is equal to its
`finrank`. -/
theorem finrank_eq_card_basis' [FiniteDimensional K V] {ι : Type w} (h : Basis ι K V) :
    (finrank K V : Cardinal.{w}) = #ι :=
  Module.mk_finrank_eq_card_basis h


theorem _root_.LinearIndependent.lt_aleph0_of_finiteDimensional {ι : Type w} [FiniteDimensional K V]
    {v : ι → V} (h : LinearIndependent K v) : #ι < ℵ₀ :=
  h.lt_aleph0_of_finite


/-- If a submodule has maximal dimension in a finite dimensional space, then it is equal to the
whole space. -/
theorem _root_.Submodule.eq_top_of_finrank_eq [FiniteDimensional K V] {S : Submodule K V}
    (h : finrank K S = finrank K V) : S = ⊤ := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem S x)) (Module.finran …
    ⊢ Eq S Top.top
  -/
  haveI : IsNoetherian K V := iff_fg.2 inferInstance
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem S x)) (Module.finran …
    this : IsNoetherian K V
    ⊢ Eq S Top.top
  -/
  set bS := Basis.ofVectorSpace K S with bS_eq
  have : LinearIndependent K ((↑) : ((↑) '' Basis.ofVectorSpaceIndex K S : Set V) → V) :=
    LinearIndependent.image_subtype (f := Submodule.subtype S)
      (by simpa [bS] using bS.linearIndependent) (by simp)
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem S x)) (Module.finran …
    this✝ : IsNoetherian K V
    bS : Basis (↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x) …
    bS_eq : Eq bS (Basis.ofVectorSpace K (Subtype fun x => Membership.mem S x))
    this : LinearIndependent K Subtype.val
    ⊢ Eq S Top.top
  -/
  set b := Basis.extend this with b_eq
  -- Porting note: `letI` now uses `this` so we need to give different names
  letI i1 : Fintype (this.extend _) :=
    (LinearIndependent.set_finite_of_isNoetherian (by simpa [b] using b.linearIndependent)).fintype
  letI i2 : Fintype (((↑) : S → V) '' Basis.ofVectorSpaceIndex K S) :=
    (LinearIndependent.set_finite_of_isNoetherian this).fintype
  letI i3 : Fintype (Basis.ofVectorSpaceIndex K S) :=
    (LinearIndependent.set_finite_of_isNoetherian
      (by simpa [bS] using bS.linearIndependent)).fintype
  have : (↑) '' Basis.ofVectorSpaceIndex K S = this.extend (Set.subset_univ _) :=
    Set.eq_of_subset_of_card_le (this.subset_extend _)
      (by
        rw [Set.card_image_of_injective _ Subtype.coe_injective, ← finrank_eq_card_basis bS, ←
            finrank_eq_card_basis b, h])
  rw [← b.span_eq, b_eq, Basis.coe_extend, Subtype.range_coe, ← this, ← Submodule.coe_subtype,
    span_image]
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem S x)) (Module.finran …
    this✝¹ : IsNoetherian K V
    bS : Basis (↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x) …
    bS_eq : Eq bS (Basis.ofVectorSpace K (Subtype fun x => Membership.mem S x))
    this✝ : LinearIndependent K Subtype.val
    b : Basis (↑(this✝.extend ⋯)) K V := Basis.extend this✝
    b_eq : Eq b (Basis.extend this✝)
    i1 : Fintype ↑(this✝.extend ⋯) := ⋯.fintype
    i2 : Fintype ↑(Set.image Subtype.val (Basis.ofVectorSpaceIndex K (Subtype fun  …
    i3 : Fintype ↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x …
    this : Eq (Set.image Subtype.val (Basis.ofVectorSpaceIndex K (Subtype fun x => …
    ⊢ Eq S (Submodule.map S.subtype (Submodule.span K (Basis.ofVectorSpaceIndex K  …
  -/
  have := bS.span_eq
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem S x)) (Module.finran …
    this✝² : IsNoetherian K V
    bS : Basis (↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x) …
    bS_eq : Eq bS (Basis.ofVectorSpace K (Subtype fun x => Membership.mem S x))
    this✝¹ : LinearIndependent K Subtype.val
    b : Basis (↑(this✝¹.extend ⋯)) K V := Basis.extend this✝¹
    b_eq : Eq b (Basis.extend this✝¹)
    i1 : Fintype ↑(this✝¹.extend ⋯) := ⋯.fintype
    i2 : Fintype ↑(Set.image Subtype.val (Basis.ofVectorSpaceIndex K (Subtype fun  …
    i3 : Fintype ↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x …
    this✝ : Eq (Set.image Subtype.val (Basis.ofVectorSpaceIndex K (Subtype fun x = …
    this : Eq (Submodule.span K (Set.range ⇑bS)) Top.top
    ⊢ Eq S (Submodule.map S.subtype (Submodule.span K (Basis.ofVectorSpaceIndex K  …
  -/
  rw [bS_eq, Basis.coe_ofVectorSpace, Subtype.range_coe] at this
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    S : Submodule K V
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem S x)) (Module.finran …
    this✝² : IsNoetherian K V
    bS : Basis (↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x) …
    bS_eq : Eq bS (Basis.ofVectorSpace K (Subtype fun x => Membership.mem S x))
    this✝¹ : LinearIndependent K Subtype.val
    b : Basis (↑(this✝¹.extend ⋯)) K V := Basis.extend this✝¹
    b_eq : Eq b (Basis.extend this✝¹)
    i1 : Fintype ↑(this✝¹.extend ⋯) := ⋯.fintype
    i2 : Fintype ↑(Set.image Subtype.val (Basis.ofVectorSpaceIndex K (Subtype fun  …
    i3 : Fintype ↑(Basis.ofVectorSpaceIndex K (Subtype fun x => Membership.mem S x …
    this✝ : Eq (Set.image Subtype.val (Basis.ofVectorSpaceIndex K (Subtype fun x = …
    this : Eq (Submodule.span K (Basis.ofVectorSpaceIndex K (Subtype fun x => Memb …
    ⊢ Eq S (Submodule.map S.subtype (Submodule.span K (Basis.ofVectorSpaceIndex K  …
  -/
  rw [this, Submodule.map_top (Submodule.subtype S), range_subtype]
  /-
    🎉 no goals
  -/


instance finiteDimensional_self : FiniteDimensional K K := inferInstance


/-- The submodule generated by a finite set is finite-dimensional. -/
theorem span_of_finite {A : Set V} (hA : Set.Finite A) : FiniteDimensional K (Submodule.span K A) :=
  Module.Finite.span_of_finite K hA


/-- The submodule generated by a single element is finite-dimensional. -/
instance span_singleton (x : V) : FiniteDimensional K (K ∙ x) :=
  Module.Finite.span_singleton K x


/-- The submodule generated by a finset is finite-dimensional. -/
instance span_finset (s : Finset V) : FiniteDimensional K (span K (s : Set V)) :=
  Module.Finite.span_finset K s


/-- Pushforwards of finite-dimensional submodules are finite-dimensional. -/
instance (f : V →ₗ[K] V₂) (p : Submodule K V) [FiniteDimensional K p] :
    FiniteDimensional K (p.map f) :=
  Module.Finite.map _ _


/-- A slight strengthening of `exists_nontrivial_relation_sum_zero_of_rank_succ_lt_card`
available when working over an ordered field:
we can ensure a positive coefficient, not just a nonzero coefficient.
-/
theorem exists_relation_sum_zero_pos_coefficient_of_finrank_succ_lt_card [FiniteDimensional L W]
    {t : Finset W} (h : finrank L W + 1 < t.card) :
    ∃ f : W → L, ∑ e ∈ t, f e • e = 0 ∧ ∑ e ∈ t, f e = 0 ∧ ∃ x ∈ t, 0 < f x := by
  obtain ⟨f, sum, total, nonzero⟩ :=
    Module.exists_nontrivial_relation_sum_zero_of_finrank_succ_lt_card h
  /-
    case intro.intro.intro
    L : Type u_1
    inst✝³ : LinearOrderedField L
    W : Type v
    inst✝² : AddCommGroup W
    inst✝¹ : Module L W
    inst✝ : FiniteDimensional L W
    t : Finset W
    h : LT.lt (HAdd.hAdd (Module.finrank L W) 1) t.card
    f : W → L
    sum : Eq (t.sum fun e => HSMul.hSMul (f e) e) 0
    total : Eq (t.sum fun e => f e) 0
    nonzero : Exists fun x => And (Membership.mem t x) (Ne (f x) 0)
    ⊢ Exists fun f => And (Eq (t.sum fun e => HSMul.hSMul (f e) e) 0) (And (Eq (t. …
  -/
  exact ⟨f, sum, total, exists_pos_of_sum_zero_of_exists_nonzero f total nonzero⟩
  /-
    🎉 no goals
  -/



/-- In a vector space with dimension 1, each set {v} is a basis for `v ≠ 0`. -/
@[simps repr_apply]
noncomputable def basisSingleton (ι : Type*) [Unique ι] (h : finrank K V = 1) (v : V)
    (hv : v ≠ 0) : Basis ι K V :=
  let b := Module.basisUnique ι h
  let h : b.repr v default ≠ 0 := mt Module.basisUnique_repr_eq_zero_iff.mp hv
  Basis.ofRepr
    { toFun := fun w => Finsupp.single default (b.repr w default / b.repr v default)
      invFun := fun f => f default • v
                     /-
                       K : Type u
                       V : Type v
                       inst✝⁵ : DivisionRing K
                       inst✝⁴ : AddCommGroup V
                       inst✝³ : Module K V
                       V₂ : Type v'
                       inst✝² : AddCommGroup V₂
                       inst✝¹ : Module K V₂
                       ι : Type u_1
                       inst✝ : Unique ι
                       h✝ : Eq (Module.finrank K V) 1
                       v : V
                       hv : Ne v 0
                       b : Basis ι K V := Module.basisUnique ι h✝
                       h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
                       ⊢ ∀ (x y : V), Eq ((fun w => Finsupp.single Inhabited.default (HDiv.hDiv ((b.r …
                     -/
      map_add' := by simp [add_div]
                     /-
                       🎉 no goals
                     -/
                      /-
                        K : Type u
                        V : Type v
                        inst✝⁵ : DivisionRing K
                        inst✝⁴ : AddCommGroup V
                        inst✝³ : Module K V
                        V₂ : Type v'
                        inst✝² : AddCommGroup V₂
                        inst✝¹ : Module K V₂
                        ι : Type u_1
                        inst✝ : Unique ι
                        h✝ : Eq (Module.finrank K V) 1
                        v : V
                        hv : Ne v 0
                        b : Basis ι K V := Module.basisUnique ι h✝
                        h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
                        ⊢ ∀ (m : K) (x : V), Eq ({ toFun := fun w => Finsupp.single Inhabited.default  …
                      -/
      map_smul' := by simp [mul_div]
                      /-
                        🎉 no goals
                      -/
      left_inv := fun w => by
        /-
          K : Type u
          V : Type v
          inst✝⁵ : DivisionRing K
          inst✝⁴ : AddCommGroup V
          inst✝³ : Module K V
          V₂ : Type v'
          inst✝² : AddCommGroup V₂
          inst✝¹ : Module K V₂
          ι : Type u_1
          inst✝ : Unique ι
          h✝ : Eq (Module.finrank K V) 1
          v : V
          hv : Ne v 0
          b : Basis ι K V := Module.basisUnique ι h✝
          h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
          w : V
          ⊢ Eq ((fun f => HSMul.hSMul (f Inhabited.default) v) ({ toFun := fun w => Fins …
        -/
        apply_fun b.repr using b.repr.toEquiv.injective
        /-
          K : Type u
          V : Type v
          inst✝⁵ : DivisionRing K
          inst✝⁴ : AddCommGroup V
          inst✝³ : Module K V
          V₂ : Type v'
          inst✝² : AddCommGroup V₂
          inst✝¹ : Module K V₂
          ι : Type u_1
          inst✝ : Unique ι
          h✝ : Eq (Module.finrank K V) 1
          v : V
          hv : Ne v 0
          b : Basis ι K V := Module.basisUnique ι h✝
          h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
          w : V
          ⊢ Eq (b.repr ((fun f => HSMul.hSMul (f Inhabited.default) v) ({ toFun := fun w …
        -/
        apply_fun Equiv.finsuppUnique
        simp only [LinearEquiv.map_smulₛₗ, Finsupp.coe_smul, Finsupp.single_eq_same,
          smul_eq_mul, Pi.smul_apply, Equiv.finsuppUnique_apply]
        /-
          K : Type u
          V : Type v
          inst✝⁵ : DivisionRing K
          inst✝⁴ : AddCommGroup V
          inst✝³ : Module K V
          V₂ : Type v'
          inst✝² : AddCommGroup V₂
          inst✝¹ : Module K V₂
          ι : Type u_1
          inst✝ : Unique ι
          h✝ : Eq (Module.finrank K V) 1
          v : V
          hv : Ne v 0
          b : Basis ι K V := Module.basisUnique ι h✝
          h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
          w : V
          ⊢ Eq (HMul.hMul ((RingHom.id K) (HDiv.hDiv ((b.repr w) Inhabited.default) ((b. …
        -/
        exact div_mul_cancel₀ _ h
        /-
          🎉 no goals
        -/
      right_inv := fun f => by
        /-
          K : Type u
          V : Type v
          inst✝⁵ : DivisionRing K
          inst✝⁴ : AddCommGroup V
          inst✝³ : Module K V
          V₂ : Type v'
          inst✝² : AddCommGroup V₂
          inst✝¹ : Module K V₂
          ι : Type u_1
          inst✝ : Unique ι
          h✝ : Eq (Module.finrank K V) 1
          v : V
          hv : Ne v 0
          b : Basis ι K V := Module.basisUnique ι h✝
          h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
          f : Finsupp ι K
          ⊢ Eq ({ toFun := fun w => Finsupp.single Inhabited.default (HDiv.hDiv ((b.repr …
        -/
        ext
        simp only [LinearEquiv.map_smulₛₗ, Finsupp.coe_smul, Finsupp.single_eq_same,
          RingHom.id_apply, smul_eq_mul, Pi.smul_apply]
        /-
          case h
          K : Type u
          V : Type v
          inst✝⁵ : DivisionRing K
          inst✝⁴ : AddCommGroup V
          inst✝³ : Module K V
          V₂ : Type v'
          inst✝² : AddCommGroup V₂
          inst✝¹ : Module K V₂
          ι : Type u_1
          inst✝ : Unique ι
          h✝ : Eq (Module.finrank K V) 1
          v : V
          hv : Ne v 0
          b : Basis ι K V := Module.basisUnique ι h✝
          h : Ne ((b.repr v) Inhabited.default) 0 := mt Module.basisUnique_repr_eq_zero_ …
          f : Finsupp ι K
          ⊢ Eq (HDiv.hDiv (HMul.hMul (f Inhabited.default) ((b.repr v) Inhabited.default …
        -/
        exact mul_div_cancel_right₀ _ h }
        /-
          🎉 no goals
        -/


@[simp]
theorem basisSingleton_apply (ι : Type*) [Unique ι] (h : finrank K V = 1) (v : V) (hv : v ≠ 0)
    (i : ι) : basisSingleton ι h v hv i = v := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    ι : Type u_1
    inst✝ : Unique ι
    h : Eq (Module.finrank K V) 1
    v : V
    hv : Ne v 0
    i : ι
    ⊢ Eq ((FiniteDimensional.basisSingleton ι h v hv) i) v
  -/
  cases Unique.uniq ‹Unique ι› i
  /-
    case refl
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    ι : Type u_1
    inst✝ : Unique ι
    h : Eq (Module.finrank K V) 1
    v : V
    hv : Ne v 0
    ⊢ Eq ((FiniteDimensional.basisSingleton ι h v hv) Inhabited.default) v
  -/
  simp [basisSingleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_basisSingleton (ι : Type*) [Unique ι] (h : finrank K V = 1) (v : V) (hv : v ≠ 0) :
                                                    /-
                                                      K : Type u
                                                      V : Type v
                                                      inst✝³ : DivisionRing K
                                                      inst✝² : AddCommGroup V
                                                      inst✝¹ : Module K V
                                                      ι : Type u_1
                                                      inst✝ : Unique ι
                                                      h : Eq (Module.finrank K V) 1
                                                      v : V
                                                      hv : Ne v 0
                                                      ⊢ Eq (Set.range ⇑(FiniteDimensional.basisSingleton ι h v hv)) (Singleton.singl …
                                                    -/
    Set.range (basisSingleton ι h v hv) = {v} := by rw [Set.range_unique, basisSingleton_apply]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem trans [FiniteDimensional F K] [FiniteDimensional K A] : FiniteDimensional F A :=
  Module.Finite.trans K A


theorem FiniteDimensional.of_rank_eq_nat {n : ℕ} (h : Module.rank K V = n) :
    FiniteDimensional K V :=
  Module.finite_of_rank_eq_nat h


@[deprecated (since := "2024-02-02")]
alias finiteDimensional_of_rank_eq_nat := FiniteDimensional.of_rank_eq_nat


theorem FiniteDimensional.of_rank_eq_zero (h : Module.rank K V = 0) : FiniteDimensional K V :=
  Module.finite_of_rank_eq_zero h


@[deprecated (since := "2024-02-02")]
alias finiteDimensional_of_rank_eq_zero := FiniteDimensional.of_rank_eq_zero


theorem FiniteDimensional.of_rank_eq_one (h : Module.rank K V = 1) : FiniteDimensional K V :=
  Module.finite_of_rank_eq_one h


@[deprecated (since := "2024-02-02")]
alias finiteDimensional_of_rank_eq_one := FiniteDimensional.of_rank_eq_one


instance finiteDimensional_bot : FiniteDimensional K (⊥ : Submodule K V) :=
                         /-
                           K : Type u
                           V : Type v
                           inst✝² : DivisionRing K
                           inst✝¹ : AddCommGroup V
                           inst✝ : Module K V
                           ⊢ Eq (Module.rank K (Subtype fun x => Membership.mem Bot.bot x)) 0
                         -/
  .of_rank_eq_zero <| by simp
                         /-
                           🎉 no goals
                         -/


/-- A submodule is finitely generated if and only if it is finite-dimensional -/
theorem fg_iff_finiteDimensional (s : Submodule K V) : s.FG ↔ FiniteDimensional K s :=
  ⟨fun h => Module.finite_def.2 <| (fg_top s).2 h, fun h => (fg_top s).1 <| Module.finite_def.1 h⟩


/-- A submodule contained in a finite-dimensional submodule is
finite-dimensional. -/
theorem finiteDimensional_of_le {S₁ S₂ : Submodule K V} [FiniteDimensional K S₂] (h : S₁ ≤ S₂) :
    FiniteDimensional K S₁ :=
  haveI : IsNoetherian K S₂ := iff_fg.2 inferInstance
  iff_fg.1
    (IsNoetherian.iff_rank_lt_aleph0.2 ((Submodule.rank_mono h).trans_lt (rank_lt_aleph0 K S₂)))


/-- The inf of two submodules, the first finite-dimensional, is
finite-dimensional. -/
instance finiteDimensional_inf_left (S₁ S₂ : Submodule K V) [FiniteDimensional K S₁] :
    FiniteDimensional K (S₁ ⊓ S₂ : Submodule K V) :=
  finiteDimensional_of_le inf_le_left


/-- The inf of two submodules, the second finite-dimensional, is
finite-dimensional. -/
instance finiteDimensional_inf_right (S₁ S₂ : Submodule K V) [FiniteDimensional K S₂] :
    FiniteDimensional K (S₁ ⊓ S₂ : Submodule K V) :=
  finiteDimensional_of_le inf_le_right


/-- The sup of two finite-dimensional submodules is
finite-dimensional. -/
instance finiteDimensional_sup (S₁ S₂ : Submodule K V) [h₁ : FiniteDimensional K S₁]
    [h₂ : FiniteDimensional K S₂] : FiniteDimensional K (S₁ ⊔ S₂ : Submodule K V) := by
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S₁ S₂ : Submodule K V
    h₁ : FiniteDimensional K (Subtype fun x => Membership.mem S₁ x)
    h₂ : FiniteDimensional K (Subtype fun x => Membership.mem S₂ x)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Max.max S₁ S₂) x)
  -/
  unfold FiniteDimensional at *
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S₁ S₂ : Submodule K V
    h₁ : Module.Finite K (Subtype fun x => Membership.mem S₁ x)
    h₂ : Module.Finite K (Subtype fun x => Membership.mem S₂ x)
    ⊢ Module.Finite K (Subtype fun x => Membership.mem (Max.max S₁ S₂) x)
  -/
  rw [finite_def] at *
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    S₁ S₂ : Submodule K V
    h₁ : Top.top.FG
    h₂ : Top.top.FG
    ⊢ Top.top.FG
  -/
  exact (fg_top _).2 (((fg_top S₁).1 h₁).sup ((fg_top S₂).1 h₂))
  /-
    🎉 no goals
  -/


/-- The submodule generated by a finite supremum of finite dimensional submodules is
finite-dimensional.

Note that strictly this only needs `∀ i ∈ s, FiniteDimensional K (S i)`, but that doesn't
work well with typeclass search. -/
instance finiteDimensional_finset_sup {ι : Type*} (s : Finset ι) (S : ι → Submodule K V)
    [∀ i, FiniteDimensional K (S i)] : FiniteDimensional K (s.sup S : Submodule K V) := by
  refine
    @Finset.sup_induction _ _ _ _ s S (fun i => FiniteDimensional K ↑i) (finiteDimensional_bot K V)
      ?_ fun i _ => by infer_instance
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    ι : Type u_1
    s : Finset ι
    S : ι → Submodule K V
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (S i) x)
    ⊢ ∀ (a₁ : Submodule K V), (fun i => FiniteDimensional K (Subtype fun x => Memb …
  -/
  intro S₁ hS₁ S₂ hS₂
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    ι : Type u_1
    s : Finset ι
    S : ι → Submodule K V
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (S i) x)
    S₁ : Submodule K V
    hS₁ : FiniteDimensional K (Subtype fun x => Membership.mem S₁ x)
    S₂ : Submodule K V
    hS₂ : FiniteDimensional K (Subtype fun x => Membership.mem S₂ x)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Max.max S₁ S₂) x)
  -/
  exact Submodule.finiteDimensional_sup S₁ S₂
  /-
    🎉 no goals
  -/


/-- The submodule generated by a supremum of finite dimensional submodules, indexed by a finite
sort is finite-dimensional. -/
instance finiteDimensional_iSup {ι : Sort*} [Finite ι] (S : ι → Submodule K V)
    [∀ i, FiniteDimensional K (S i)] : FiniteDimensional K ↑(⨆ i, S i) := by
  /-
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Sort u_1
    inst✝¹ : Finite ι
    S : ι → Submodule K V
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (S i) x)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => S i) x)
  -/
  cases nonempty_fintype (PLift ι)
  /-
    case intro
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Sort u_1
    inst✝¹ : Finite ι
    S : ι → Submodule K V
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (S i) x)
    val✝ : Fintype (PLift ι)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (iSup fun i => S i) x)
  -/
  rw [← iSup_plift_down, ← Finset.sup_univ_eq_iSup]
  /-
    case intro
    K : Type u
    V : Type v
    inst✝⁴ : DivisionRing K
    inst✝³ : AddCommGroup V
    inst✝² : Module K V
    ι : Sort u_1
    inst✝¹ : Finite ι
    S : ι → Submodule K V
    inst✝ : ∀ (i : ι), FiniteDimensional K (Subtype fun x => Membership.mem (S i) x)
    val✝ : Fintype (PLift ι)
    ⊢ FiniteDimensional K (Subtype fun x => Membership.mem (Finset.univ.sup fun i  …
  -/
  exact Submodule.finiteDimensional_finset_sup _ _
  /-
    🎉 no goals
  -/


/-- Finite dimensionality is preserved under linear equivalence. -/
protected theorem finiteDimensional (f : V ≃ₗ[K] V₂) [FiniteDimensional K V] :
    FiniteDimensional K V₂ :=
  Module.Finite.equiv f


instance finiteDimensional_finsupp {ι : Type*} [Finite ι] [FiniteDimensional K V] :
    FiniteDimensional K (ι →₀ V) :=
  Module.Finite.finsupp


/-- If a submodule is contained in a finite-dimensional
submodule with the same or smaller dimension, they are equal. -/
theorem eq_of_le_of_finrank_le {S₁ S₂ : Submodule K V} [FiniteDimensional K S₂] (hle : S₁ ≤ S₂)
    (hd : finrank K S₂ ≤ finrank K S₁) : S₁ = S₂ := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    S₁ S₂ : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem S₂ x)
    hle : LE.le S₁ S₂
    hd : LE.le (Module.finrank K (Subtype fun x => Membership.mem S₂ x)) (Module.f …
    ⊢ Eq S₁ S₂
  -/
  rw [← LinearEquiv.finrank_eq (Submodule.comapSubtypeEquivOfLe hle)] at hd
  exact le_antisymm hle (Submodule.comap_subtype_eq_top.1
    (eq_top_of_finrank_eq (le_antisymm (comap (Submodule.subtype S₂) S₁).finrank_le hd)))


/-- If a submodule is contained in a finite-dimensional
submodule with the same dimension, they are equal. -/
theorem eq_of_le_of_finrank_eq {S₁ S₂ : Submodule K V} [FiniteDimensional K S₂] (hle : S₁ ≤ S₂)
    (hd : finrank K S₁ = finrank K S₂) : S₁ = S₂ :=
  eq_of_le_of_finrank_le hle hd.ge


/-- If a subalgebra is contained in a finite-dimensional
subalgebra with the same or smaller dimension, they are equal. -/
theorem eq_of_le_of_finrank_le (h_le : F ≤ E) (h_finrank : finrank K E ≤ finrank K F) : F = E :=
  haveI : Module.Finite K (Subalgebra.toSubmodule E) := hfin
  toSubmodule_injective <| Submodule.eq_of_le_of_finrank_le h_le h_finrank


/-- If a subalgebra is contained in a finite-dimensional
subalgebra with the same dimension, they are equal. -/
theorem eq_of_le_of_finrank_eq (h_le : F ≤ E) (h_finrank : finrank K F = finrank K E) : F = E :=
  eq_of_le_of_finrank_le h_le h_finrank.ge


/-- On a finite-dimensional space, an injective linear map is surjective. -/
theorem surjective_of_injective [FiniteDimensional K V] {f : V →ₗ[K] V} (hinj : Injective f) :
    Surjective f := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V
    hinj : Function.Injective ⇑f
    ⊢ Function.Surjective ⇑f
  -/
  have h := rank_range_of_injective _ hinj
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V
    hinj : Function.Injective ⇑f
    h : Eq (Module.rank K (Subtype fun x => Membership.mem (LinearMap.range f) x)) …
    ⊢ Function.Surjective ⇑f
  -/
  rw [← finrank_eq_rank, ← finrank_eq_rank, Nat.cast_inj] at h
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V
    hinj : Function.Injective ⇑f
    h : Eq (Module.finrank K (Subtype fun x => Membership.mem (LinearMap.range f)  …
    ⊢ Function.Surjective ⇑f
  -/
  exact range_eq_top.1 (eq_top_of_finrank_eq h)
  /-
    🎉 no goals
  -/


/-- The image under an onto linear map of a finite-dimensional space is also finite-dimensional. -/
theorem finiteDimensional_of_surjective [FiniteDimensional K V] (f : V →ₗ[K] V₂)
    (hf : LinearMap.range f = ⊤) : FiniteDimensional K V₂ :=
  Module.Finite.of_surjective f <| range_eq_top.1 hf


/-- The range of a linear map defined on a finite-dimensional space is also finite-dimensional. -/
instance finiteDimensional_range [FiniteDimensional K V] (f : V →ₗ[K] V₂) :
    FiniteDimensional K (LinearMap.range f) :=
  Module.Finite.range f


/-- On a finite-dimensional space, a linear map is injective if and only if it is surjective. -/
theorem injective_iff_surjective [FiniteDimensional K V] {f : V →ₗ[K] V} :
    Injective f ↔ Surjective f :=
  ⟨surjective_of_injective, fun hsurj =>
    let ⟨g, hg⟩ := f.exists_rightInverse_of_surjective (range_eq_top.2 hsurj)
    have : Function.RightInverse g f := LinearMap.ext_iff.1 hg
    (leftInverse_of_surjective_of_rightInverse (surjective_of_injective this.injective)
        this).injective⟩


lemma injOn_iff_surjOn {p : Submodule K V} [FiniteDimensional K p]
    {f : V →ₗ[K] V} (h : ∀ x ∈ p, f x ∈ p) :
    Set.InjOn f p ↔ Set.SurjOn f p p := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    p : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem p x)
    f : LinearMap (RingHom.id K) V V
    h : ∀ (x : V), Membership.mem p x → Membership.mem p (f x)
    ⊢ Iff (Set.InjOn ⇑f ↑p) (Set.SurjOn ⇑f ↑p ↑p)
  -/
  rw [Set.injOn_iff_injective, ← Set.MapsTo.restrict_surjective_iff h]
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    p : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem p x)
    f : LinearMap (RingHom.id K) V V
    h : ∀ (x : V), Membership.mem p x → Membership.mem p (f x)
    ⊢ Iff (Function.Injective ((↑p).restrict ⇑f)) (Function.Surjective (Set.MapsTo …
  -/
  change Injective (f.domRestrict p) ↔ Surjective (f.restrict h)
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    p : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem p x)
    f : LinearMap (RingHom.id K) V V
    h : ∀ (x : V), Membership.mem p x → Membership.mem p (f x)
    ⊢ Iff (Function.Injective ⇑(f.domRestrict p)) (Function.Surjective ⇑(f.restric …
  -/
  simp [disjoint_iff, ← injective_iff_surjective]
  /-
    🎉 no goals
  -/


theorem ker_eq_bot_iff_range_eq_top [FiniteDimensional K V] {f : V →ₗ[K] V} :
    LinearMap.ker f = ⊥ ↔ LinearMap.range f = ⊤ := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V
    ⊢ Iff (Eq (LinearMap.ker f) Bot.bot) (Eq (LinearMap.range f) Top.top)
  -/
  rw [range_eq_top, ker_eq_bot, injective_iff_surjective]
  /-
    🎉 no goals
  -/


/-- In a finite-dimensional space, if linear maps are inverse to each other on one side then they
are also inverse to each other on the other side. -/
theorem mul_eq_one_of_mul_eq_one [FiniteDimensional K V] {f g : V →ₗ[K] V} (hfg : f * g = 1) :
    g * f = 1 := by
  have ginj : Injective g :=
    HasLeftInverse.injective ⟨f, fun x => show (f * g) x = (1 : V →ₗ[K] V) x by rw [hfg]⟩
  let ⟨i, hi⟩ := g.exists_rightInverse_of_surjective
    (range_eq_top.2 (injective_iff_surjective.1 ginj))
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f g : LinearMap (RingHom.id K) V V
    hfg : Eq (HMul.hMul f g) 1
    ginj : Function.Injective ⇑g
    i : LinearMap (RingHom.id K) V V
    hi : Eq (g.comp i) LinearMap.id
    ⊢ Eq (HMul.hMul g f) 1
  -/
  have : f * (g * i) = f * 1 := congr_arg _ hi
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f g : LinearMap (RingHom.id K) V V
    hfg : Eq (HMul.hMul f g) 1
    ginj : Function.Injective ⇑g
    i : LinearMap (RingHom.id K) V V
    hi : Eq (g.comp i) LinearMap.id
    this : Eq (HMul.hMul f (HMul.hMul g i)) (HMul.hMul f 1)
    ⊢ Eq (HMul.hMul g f) 1
  -/
  rw [← mul_assoc, hfg, one_mul, mul_one] at this; rwa [← this]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- In a finite-dimensional space, linear maps are inverse to each other on one side if and only if
they are inverse to each other on the other side. -/
theorem mul_eq_one_comm [FiniteDimensional K V] {f g : V →ₗ[K] V} : f * g = 1 ↔ g * f = 1 :=
  ⟨mul_eq_one_of_mul_eq_one, mul_eq_one_of_mul_eq_one⟩


/-- In a finite-dimensional space, linear maps are inverse to each other on one side if and only if
they are inverse to each other on the other side. -/
theorem comp_eq_id_comm [FiniteDimensional K V] {f g : V →ₗ[K] V} : f.comp g = id ↔ g.comp f = id :=
  mul_eq_one_comm


theorem comap_eq_sup_ker_of_disjoint {p : Submodule K V} [FiniteDimensional K p] {f : V →ₗ[K] V}
    (h : ∀ x ∈ p, f x ∈ p) (h' : Disjoint p (ker f)) :
    p.comap f = p ⊔ ker f := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    p : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem p x)
    f : LinearMap (RingHom.id K) V V
    h : ∀ (x : V), Membership.mem p x → Membership.mem p (f x)
    h' : Disjoint p (LinearMap.ker f)
    ⊢ Eq (Submodule.comap f p) (Max.max p (LinearMap.ker f))
  -/
  refine le_antisymm (fun x hx ↦ ?_) (sup_le_iff.mpr ⟨h, ker_le_comap _⟩)
  obtain ⟨⟨y, hy⟩, hxy⟩ :=
    surjective_of_injective ((injective_restrict_iff_disjoint h).mpr h') ⟨f x, hx⟩
  /-
    case intro.mk
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    p : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem p x)
    f : LinearMap (RingHom.id K) V V
    h : ∀ (x : V), Membership.mem p x → Membership.mem p (f x)
    h' : Disjoint p (LinearMap.ker f)
    x : V
    hx : Membership.mem (Submodule.comap f p) x
    y : V
    hy : Membership.mem p y
    hxy : Eq ((f.restrict h) ⟨y, hy⟩) ⟨f x, hx⟩
    ⊢ Membership.mem (Max.max p (LinearMap.ker f)) x
  -/
  replace hxy : f y = f x := by simpa [Subtype.ext_iff] using hxy
  /-
    case intro.mk
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    p : Submodule K V
    inst✝ : FiniteDimensional K (Subtype fun x => Membership.mem p x)
    f : LinearMap (RingHom.id K) V V
    h : ∀ (x : V), Membership.mem p x → Membership.mem p (f x)
    h' : Disjoint p (LinearMap.ker f)
    x : V
    hx : Membership.mem (Submodule.comap f p) x
    y : V
    hy : Membership.mem p y
    hxy : Eq (f y) (f x)
    ⊢ Membership.mem (Max.max p (LinearMap.ker f)) x
  -/
  exact Submodule.mem_sup.mpr ⟨y, hy, x - y, by simp [hxy], add_sub_cancel y x⟩
  /-
    🎉 no goals
  -/


theorem ker_comp_eq_of_commute_of_disjoint_ker [FiniteDimensional K V] {f g : V →ₗ[K] V}
    (h : Commute f g) (h' : Disjoint (ker f) (ker g)) :
    ker (f ∘ₗ g) = ker f ⊔ ker g := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f g : LinearMap (RingHom.id K) V V
    h : Commute f g
    h' : Disjoint (LinearMap.ker f) (LinearMap.ker g)
    ⊢ Eq (LinearMap.ker (f.comp g)) (Max.max (LinearMap.ker f) (LinearMap.ker g))
  -/
  suffices ∀ x, f x = 0 → f (g x) = 0 by rw [ker_comp, comap_eq_sup_ker_of_disjoint _ h']; simpa
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f g : LinearMap (RingHom.id K) V V
    h : Commute f g
    h' : Disjoint (LinearMap.ker f) (LinearMap.ker g)
    ⊢ ∀ (x : V), Eq (f x) 0 → Eq (f (g x)) 0
  -/
  intro x hx
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f g : LinearMap (RingHom.id K) V V
    h : Commute f g
    h' : Disjoint (LinearMap.ker f) (LinearMap.ker g)
    x : V
    hx : Eq (f x) 0
    ⊢ Eq (f (g x)) 0
  -/
  rw [← comp_apply, ← mul_eq_comp, h.eq, mul_apply, hx, _root_.map_zero]
  /-
    🎉 no goals
  -/


theorem ker_noncommProd_eq_of_supIndep_ker [FiniteDimensional K V] {ι : Type*} {f : ι → V →ₗ[K] V}
    (s : Finset ι) (comm) (h : s.SupIndep fun i ↦ ker (f i)) :
    ker (s.noncommProd f comm) = ⨆ i ∈ s, ker (f i) := by
  classical
  induction' s using Finset.induction_on with i s hi ih
  · set_option tactic.skipAssignedInstances false in
    simpa using LinearMap.ker_id
  replace ih : ker (Finset.noncommProd s f <| Set.Pairwise.mono (s.subset_insert i) comm) =
      ⨆ x ∈ s, ker (f x) := ih _ (h.subset (s.subset_insert i))
  rw [Finset.noncommProd_insert_of_not_mem _ _ _ _ hi, mul_eq_comp,
    ker_comp_eq_of_commute_of_disjoint_ker]
  · simp_rw [Finset.mem_insert_coe, iSup_insert, Finset.mem_coe, ih]
  · exact s.noncommProd_commute _ _ _ fun j hj ↦
      comm (s.mem_insert_self i) (Finset.mem_insert_of_mem hj) (by aesop)
  · replace h := Finset.supIndep_iff_disjoint_erase.mp h i (s.mem_insert_self i)
    simpa [ih, hi, Finset.sup_eq_iSup] using h


/-- The linear equivalence corresponding to an injective endomorphism. -/
noncomputable def ofInjectiveEndo (f : V →ₗ[K] V) (h_inj : Injective f) : V ≃ₗ[K] V :=
  LinearEquiv.ofBijective f ⟨h_inj, LinearMap.injective_iff_surjective.mp h_inj⟩


@[simp]
theorem coe_ofInjectiveEndo (f : V →ₗ[K] V) (h_inj : Injective f) :
    ⇑(ofInjectiveEndo f h_inj) = f :=
  rfl


@[simp]
theorem ofInjectiveEndo_right_inv (f : V →ₗ[K] V) (h_inj : Injective f) :
    f * (ofInjectiveEndo f h_inj).symm = 1 :=
  LinearMap.ext <| (ofInjectiveEndo f h_inj).apply_symm_apply


@[simp]
theorem ofInjectiveEndo_left_inv (f : V →ₗ[K] V) (h_inj : Injective f) :
    ((ofInjectiveEndo f h_inj).symm : V →ₗ[K] V) * f = 1 :=
  LinearMap.ext <| (ofInjectiveEndo f h_inj).symm_apply_apply


theorem isUnit_iff_ker_eq_bot [FiniteDimensional K V] (f : V →ₗ[K] V) :
    IsUnit f ↔ (LinearMap.ker f) = ⊥ := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V
    ⊢ Iff (IsUnit f) (Eq (LinearMap.ker f) Bot.bot)
  -/
  constructor
    /-
      case mp
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : LinearMap (RingHom.id K) V V
      ⊢ IsUnit f → Eq (LinearMap.ker f) Bot.bot
    -/
  · rintro ⟨u, rfl⟩
    /-
      case mp.intro
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      u : Units (LinearMap (RingHom.id K) V V)
      ⊢ Eq (LinearMap.ker ↑u) Bot.bot
    -/
    exact LinearMap.ker_eq_bot_of_inverse u.inv_mul
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : LinearMap (RingHom.id K) V V
      ⊢ Eq (LinearMap.ker f) Bot.bot → IsUnit f
    -/
  · intro h_inj
    /-
      case mpr
      K : Type u
      V : Type v
      inst✝³ : DivisionRing K
      inst✝² : AddCommGroup V
      inst✝¹ : Module K V
      inst✝ : FiniteDimensional K V
      f : LinearMap (RingHom.id K) V V
      h_inj : Eq (LinearMap.ker f) Bot.bot
      ⊢ IsUnit f
    -/
    rw [ker_eq_bot] at h_inj
    exact ⟨⟨f, (LinearEquiv.ofInjectiveEndo f h_inj).symm.toLinearMap,
      LinearEquiv.ofInjectiveEndo_right_inv f h_inj, LinearEquiv.ofInjectiveEndo_left_inv f h_inj⟩,
      rfl⟩


theorem isUnit_iff_range_eq_top [FiniteDimensional K V] (f : V →ₗ[K] V) :
    IsUnit f ↔ (LinearMap.range f) = ⊤ := by
  /-
    K : Type u
    V : Type v
    inst✝³ : DivisionRing K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : LinearMap (RingHom.id K) V V
    ⊢ Iff (IsUnit f) (Eq (LinearMap.range f) Top.top)
  -/
  rw [isUnit_iff_ker_eq_bot, ker_eq_bot_iff_range_eq_top]
  /-
    🎉 no goals
  -/


theorem finrank_zero_iff_forall_zero [FiniteDimensional K V] : finrank K V = 0 ↔ ∀ x : V, x = 0 :=
  Module.finrank_zero_iff.trans (subsingleton_iff_forall_eq 0)


/-- If `ι` is an empty type and `V` is zero-dimensional, there is a unique `ι`-indexed basis. -/
noncomputable def basisOfFinrankZero [FiniteDimensional K V] {ι : Type*} [IsEmpty ι]
    (hV : finrank K V = 0) : Basis ι K V :=
  haveI : Subsingleton V := finrank_zero_iff.1 hV
  Basis.empty _


lemma FiniteDimensional.exists_mul_eq_one (F : Type*) {K : Type*} [Field F] [Ring K] [IsDomain K]
    [Algebra F K] [FiniteDimensional F K] {x : K} (H : x ≠ 0) : ∃ y, x * y = 1 := by
  have : Function.Surjective (LinearMap.mulLeft F x) :=
    LinearMap.injective_iff_surjective.1 fun y z => ((mul_right_inj' H).1 : x * y = x * z → y = z)
  /-
    F : Type u_1
    K : Type u_2
    inst✝⁴ : Field F
    inst✝³ : Ring K
    inst✝² : IsDomain K
    inst✝¹ : Algebra F K
    inst✝ : FiniteDimensional F K
    x : K
    H : Ne x 0
    this : Function.Surjective ⇑(LinearMap.mulLeft F x)
    ⊢ Exists fun y => Eq (HMul.hMul x y) 1
  -/
  exact this 1
  /-
    🎉 no goals
  -/


/-- A domain that is module-finite as an algebra over a field is a division ring. -/
noncomputable def divisionRingOfFiniteDimensional (F K : Type*) [Field F] [Ring K] [IsDomain K]
    [Algebra F K] [FiniteDimensional F K] : DivisionRing K where
  __ := ‹IsDomain K›
  inv x :=
    letI := Classical.decEq K
    if H : x = 0 then 0 else Classical.choose <| FiniteDimensional.exists_mul_eq_one F H
  mul_inv_cancel x hx := show x * dite _ (h := _) _ _ = _ by
    /-
      K✝ : Type u
      V : Type v
      F : Type u_1
      K : Type u_2
      inst✝⁴ : Field F
      inst✝³ : Ring K
      inst✝² : IsDomain K
      inst✝¹ : Algebra F K
      inst✝ : FiniteDimensional F K
      x : K
      hx : Ne x 0
      ⊢ Eq (HMul.hMul x (dite (Eq x 0) (fun H => 0) fun H => Classical.choose ⋯)) 1
    -/
    rw [dif_neg hx]
    /-
      K✝ : Type u
      V : Type v
      F : Type u_1
      K : Type u_2
      inst✝⁴ : Field F
      inst✝³ : Ring K
      inst✝² : IsDomain K
      inst✝¹ : Algebra F K
      inst✝ : FiniteDimensional F K
      x : K
      hx : Ne x 0
      ⊢ Eq (HMul.hMul x (Classical.choose ⋯)) 1
    -/
    exact (Classical.choose_spec (FiniteDimensional.exists_mul_eq_one F hx) :)
    /-
      🎉 no goals
    -/
  inv_zero := dif_pos rfl
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


lemma FiniteDimensional.isUnit (F : Type*) {K : Type*} [Field F] [Ring K] [IsDomain K]
    [Algebra F K] [FiniteDimensional F K] {x : K} (H : x ≠ 0) : IsUnit x :=
  let _ := divisionRingOfFiniteDimensional F K; H.isUnit


/-- An integral domain that is module-finite as an algebra over a field is a field. -/
noncomputable def fieldOfFiniteDimensional (F K : Type*) [Field F] [h : CommRing K] [IsDomain K]
    [Algebra F K] [FiniteDimensional F K] : Field K :=
  { divisionRingOfFiniteDimensional F K with
    toCommRing := h }


theorem finrank_span_singleton {v : V} (hv : v ≠ 0) : finrank K (K ∙ v) = 1 := by
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : V
    hv : Ne v 0
    ⊢ Eq (Module.finrank K (Subtype fun x => Membership.mem (Submodule.span K (Sin …
  -/
  apply le_antisymm
    /-
      case a
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem (Submodule.span K ( …
    -/
  · exact finrank_span_le_card ({v} : Set V)
    /-
      🎉 no goals
    -/
    /-
      case a
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ LE.le 1 (Module.finrank K (Subtype fun x => Membership.mem (Submodule.span K …
    -/
  · rw [Nat.succ_le_iff, finrank_pos_iff]
    /-
      case a
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ Nontrivial (Subtype fun x => Membership.mem (Submodule.span K (Singleton.sin …
    -/
    use ⟨v, mem_span_singleton_self v⟩, 0
    /-
      case h
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ Ne ⟨v, ⋯⟩ 0
    -/
    apply Subtype.coe_ne_coe.mp
    /-
      case h
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      v : V
      hv : Ne v 0
      ⊢ Ne ↑⟨v, ⋯⟩ ↑0
    -/
    simp [hv]
    /-
      🎉 no goals
    -/


/-- In a one-dimensional space, any vector is a multiple of any nonzero vector -/
lemma exists_smul_eq_of_finrank_eq_one
    (h : finrank K V = 1) {x : V} (hx : x ≠ 0) (y : V) :
    ∃ (c : K), c • x = y := by
  have : Submodule.span K {x} = ⊤ := by
    have : FiniteDimensional K V := .of_finrank_eq_succ h
    apply eq_top_of_finrank_eq
    rw [h]
    exact finrank_span_singleton hx
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : Eq (Module.finrank K V) 1
    x : V
    hx : Ne x 0
    y : V
    this : Eq (Submodule.span K (Singleton.singleton x)) Top.top
    ⊢ Exists fun c => Eq (HSMul.hSMul c x) y
  -/
  have : y ∈ Submodule.span K {x} := by rw [this]; exact mem_top
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    h : Eq (Module.finrank K V) 1
    x : V
    hx : Ne x 0
    y : V
    this✝ : Eq (Submodule.span K (Singleton.singleton x)) Top.top
    this : Membership.mem (Submodule.span K (Singleton.singleton x)) y
    ⊢ Exists fun c => Eq (HSMul.hSMul c x) y
  -/
  exact mem_span_singleton.1 this
  /-
    🎉 no goals
  -/


theorem Set.finrank_mono [FiniteDimensional K V] {s t : Set V} (h : s ⊆ t) :
    s.finrank K ≤ t.finrank K :=
  Submodule.finrank_mono (span_mono h)


/-- A vector space with a nonzero vector `v` has dimension 1 iff `v` spans.
-/
theorem finrank_eq_one_iff_of_nonzero (v : V) (nz : v ≠ 0) :
    finrank K V = 1 ↔ span K ({v} : Set V) = ⊤ :=
               /-
                 K : Type u
                 V : Type v
                 inst✝² : DivisionRing K
                 inst✝¹ : AddCommGroup V
                 inst✝ : Module K V
                 v : V
                 nz : Ne v 0
                 h : Eq (Module.finrank K V) 1
                 ⊢ Eq (Submodule.span K (Singleton.singleton v)) Top.top
               -/
  ⟨fun h => by simpa using (basisSingleton Unit h v nz).span_eq, fun s =>
               /-
                 🎉 no goals
               -/
    finrank_eq_card_basis
      (Basis.mk (linearIndependent_singleton nz)
        (by
          /-
            K : Type u
            V : Type v
            inst✝² : DivisionRing K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            v : V
            nz : Ne v 0
            s : Eq (Submodule.span K (Singleton.singleton v)) Top.top
            ⊢ LE.le Top.top (Submodule.span K (Set.range fun x => ↑x))
          -/
          convert s.ge  -- Porting note: added `.ge` to make things easier for `convert`
          /-
            case h.e'_4.h.e'_6
            K : Type u
            V : Type v
            inst✝² : DivisionRing K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            v : V
            nz : Ne v 0
            s : Eq (Submodule.span K (Singleton.singleton v)) Top.top
            ⊢ Eq (Set.range fun x => ↑x) (Singleton.singleton v)
          -/
          simp))⟩
          /-
            🎉 no goals
          -/


/-- A module with a nonzero vector `v` has dimension 1 iff every vector is a multiple of `v`.
-/
theorem finrank_eq_one_iff_of_nonzero' (v : V) (nz : v ≠ 0) :
    finrank K V = 1 ↔ ∀ w : V, ∃ c : K, c • v = w := by
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : V
    nz : Ne v 0
    ⊢ Iff (Eq (Module.finrank K V) 1) (∀ (w : V), Exists fun c => Eq (HSMul.hSMul  …
  -/
  rw [finrank_eq_one_iff_of_nonzero v nz]
  /-
    K : Type u
    V : Type v
    inst✝² : DivisionRing K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    v : V
    nz : Ne v 0
    ⊢ Iff (Eq (Submodule.span K (Singleton.singleton v)) Top.top) (∀ (w : V), Exis …
  -/
  apply span_singleton_eq_top_iff
  /-
    🎉 no goals
  -/

-- We use the `LinearMap.CompatibleSMul` typeclass here, to encompass two situations:
-- * `A = K`
-- * `[Field K] [Algebra K A] [IsScalarTower K A V] [IsScalarTower K A W]`

theorem surjective_of_nonzero_of_finrank_eq_one {W A : Type*} [Semiring A] [Module A V]
    [AddCommGroup W] [Module K W] [Module A W] [LinearMap.CompatibleSMul V W K A]
    (h : finrank K W = 1) {f : V →ₗ[A] W} (w : f ≠ 0) : Surjective f := by
  /-
    K : Type u
    V : Type v
    inst✝⁸ : DivisionRing K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    W : Type u_1
    A : Type u_2
    inst✝⁵ : Semiring A
    inst✝⁴ : Module A V
    inst✝³ : AddCommGroup W
    inst✝² : Module K W
    inst✝¹ : Module A W
    inst✝ : LinearMap.CompatibleSMul V W K A
    h : Eq (Module.finrank K W) 1
    f : LinearMap (RingHom.id A) V W
    w : Ne f 0
    ⊢ Function.Surjective ⇑f
  -/
  change Surjective (f.restrictScalars K)
  /-
    K : Type u
    V : Type v
    inst✝⁸ : DivisionRing K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    W : Type u_1
    A : Type u_2
    inst✝⁵ : Semiring A
    inst✝⁴ : Module A V
    inst✝³ : AddCommGroup W
    inst✝² : Module K W
    inst✝¹ : Module A W
    inst✝ : LinearMap.CompatibleSMul V W K A
    h : Eq (Module.finrank K W) 1
    f : LinearMap (RingHom.id A) V W
    w : Ne f 0
    ⊢ Function.Surjective ⇑(↑K f)
  -/
  obtain ⟨v, n⟩ := DFunLike.ne_iff.mp w
  /-
    case intro
    K : Type u
    V : Type v
    inst✝⁸ : DivisionRing K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    W : Type u_1
    A : Type u_2
    inst✝⁵ : Semiring A
    inst✝⁴ : Module A V
    inst✝³ : AddCommGroup W
    inst✝² : Module K W
    inst✝¹ : Module A W
    inst✝ : LinearMap.CompatibleSMul V W K A
    h : Eq (Module.finrank K W) 1
    f : LinearMap (RingHom.id A) V W
    w : Ne f 0
    v : V
    n : Ne (f v) (0 v)
    ⊢ Function.Surjective ⇑(↑K f)
  -/
  intro z
  /-
    case intro
    K : Type u
    V : Type v
    inst✝⁸ : DivisionRing K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    W : Type u_1
    A : Type u_2
    inst✝⁵ : Semiring A
    inst✝⁴ : Module A V
    inst✝³ : AddCommGroup W
    inst✝² : Module K W
    inst✝¹ : Module A W
    inst✝ : LinearMap.CompatibleSMul V W K A
    h : Eq (Module.finrank K W) 1
    f : LinearMap (RingHom.id A) V W
    w : Ne f 0
    v : V
    n : Ne (f v) (0 v)
    z : W
    ⊢ Exists fun a => Eq ((↑K f) a) z
  -/
  obtain ⟨c, rfl⟩ := (finrank_eq_one_iff_of_nonzero' (f v) n).mp h z
  /-
    case intro.intro
    K : Type u
    V : Type v
    inst✝⁸ : DivisionRing K
    inst✝⁷ : AddCommGroup V
    inst✝⁶ : Module K V
    W : Type u_1
    A : Type u_2
    inst✝⁵ : Semiring A
    inst✝⁴ : Module A V
    inst✝³ : AddCommGroup W
    inst✝² : Module K W
    inst✝¹ : Module A W
    inst✝ : LinearMap.CompatibleSMul V W K A
    h : Eq (Module.finrank K W) 1
    f : LinearMap (RingHom.id A) V W
    w : Ne f 0
    v : V
    n : Ne (f v) (0 v)
    c : K
    ⊢ Exists fun a => Eq ((↑K f) a) (HSMul.hSMul c (f v))
  -/
  exact ⟨c • v, by simp⟩
  /-
    🎉 no goals
  -/


/-- A `Subalgebra` is `FiniteDimensional` iff it is `FiniteDimensional` as a submodule. -/
theorem Subalgebra.finiteDimensional_toSubmodule {S : Subalgebra F E} :
    FiniteDimensional F (Subalgebra.toSubmodule S) ↔ FiniteDimensional F S :=
  Iff.rfl


alias ⟨FiniteDimensional.of_subalgebra_toSubmodule, FiniteDimensional.subalgebra_toSubmodule⟩ :=
  Subalgebra.finiteDimensional_toSubmodule


instance FiniteDimensional.finiteDimensional_subalgebra [FiniteDimensional F E]
    (S : Subalgebra F E) : FiniteDimensional F S :=
  FiniteDimensional.of_subalgebra_toSubmodule inferInstance


@[deprecated Subalgebra.finite_bot (since := "2024-04-11")]
theorem Subalgebra.finiteDimensional_bot : FiniteDimensional F (⊥ : Subalgebra F E) :=
  Subalgebra.finite_bot


theorem ker_pow_constant {f : End K V} {k : ℕ}
    (h : LinearMap.ker (f ^ k) = LinearMap.ker (f ^ k.succ)) :
    ∀ m, LinearMap.ker (f ^ k) = LinearMap.ker (f ^ (k + m))
            /-
              K : Type u
              V : Type v
              inst✝² : DivisionRing K
              inst✝¹ : AddCommGroup V
              inst✝ : Module K V
              f : Module.End K V
              k : Nat
              h : Eq (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f k.succ))
              ⊢ Eq (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f (HAdd.hAdd k  …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | m + 1 => by
    /-
      K : Type u
      V : Type v
      inst✝² : DivisionRing K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      k : Nat
      h : Eq (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f k.succ))
      m : Nat
      ⊢ Eq (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f (HAdd.hAdd k  …
    -/
    apply le_antisymm
      /-
        case a
        K : Type u
        V : Type v
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : Module.End K V
        k : Nat
        h : Eq (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f k.succ))
        m : Nat
        ⊢ LE.le (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f (HAdd.hAdd …
      -/
    · rw [add_comm, pow_add]
      /-
        case a
        K : Type u
        V : Type v
        inst✝² : DivisionRing K
        inst✝¹ : AddCommGroup V
        inst✝ : Module K V
        f : Module.End K V
        k : Nat
        h : Eq (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HPow.hPow f k.succ))
        m : Nat
        ⊢ LE.le (LinearMap.ker (HPow.hPow f k)) (LinearMap.ker (HMul.hMul (HPow.hPow f …
      -/
      apply LinearMap.ker_le_ker_comp
      /-
        🎉 no goals
      -/
    · rw [ker_pow_constant h m, add_comm m 1, ← add_assoc, pow_add, pow_add f k m,
        LinearMap.mul_eq_comp, LinearMap.mul_eq_comp, LinearMap.ker_comp, LinearMap.ker_comp, h,
        Nat.add_one]


