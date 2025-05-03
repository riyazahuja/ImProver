/-- A continuous partition of unity on a set `s : Set X` is a collection of continuous functions
`f i` such that

* the supports of `f i` form a locally finite family of sets, i.e., for every point `x : X` there
  exists a neighborhood `U ∋ x` such that all but finitely many functions `f i` are zero on `U`;
* the functions `f i` are nonnegative;
* the sum `∑ᶠ i, f i x` is equal to one for every `x ∈ s` and is less than or equal to one
  otherwise.

If `X` is a normal paracompact space, then `PartitionOfUnity.exists_isSubordinate` guarantees
that for every open covering `U : Set (Set X)` of `s` there exists a partition of unity that is
subordinate to `U`.
-/
structure PartitionOfUnity (ι X : Type*) [TopologicalSpace X] (s : Set X := univ) where
  /-- The collection of continuous functions underlying this partition of unity -/
  toFun : ι → C(X, ℝ)
  /-- the supports of the underlying functions are a locally finite family of sets -/
  locallyFinite' : LocallyFinite fun i => support (toFun i)
  /-- the functions are non-negative -/
  nonneg' : 0 ≤ toFun
  /-- the functions sum up to one on `s` -/
  sum_eq_one' : ∀ x ∈ s, ∑ᶠ i, toFun i x = 1
  /-- the functions sum up to at most one, globally -/
  sum_le_one' : ∀ x, ∑ᶠ i, toFun i x ≤ 1


/-- A `BumpCovering ι X s` is an indexed family of functions `f i`, `i : ι`, such that

* the supports of `f i` form a locally finite family of sets, i.e., for every point `x : X` there
  exists a neighborhood `U ∋ x` such that all but finitely many functions `f i` are zero on `U`;
* for all `i`, `x` we have `0 ≤ f i x ≤ 1`;
* each point `x ∈ s` belongs to the interior of `{x | f i x = 1}` for some `i`.

One of the main use cases for a `BumpCovering` is to define a `PartitionOfUnity`, see
`BumpCovering.toPartitionOfUnity`, but some proofs can directly use a `BumpCovering` instead of
a `PartitionOfUnity`.

If `X` is a normal paracompact space, then `BumpCovering.exists_isSubordinate` guarantees that for
every open covering `U : Set (Set X)` of `s` there exists a `BumpCovering` of `s` that is
subordinate to `U`.
-/
structure BumpCovering (ι X : Type*) [TopologicalSpace X] (s : Set X := univ) where
  /-- The collections of continuous functions underlying this bump covering -/
  toFun : ι → C(X, ℝ)
  /-- the supports of the underlying functions are a locally finite family of sets -/
  locallyFinite' : LocallyFinite fun i => support (toFun i)
  /-- the functions are non-negative -/
  nonneg' : 0 ≤ toFun
  /-- the functions are each at most one -/
  le_one' : toFun ≤ 1
  /-- Each point `x ∈ s` belongs to the interior of `{x | f i x = 1}` for some `i`. -/
  eventuallyEq_one' : ∀ x ∈ s, ∃ i, toFun i =ᶠ[𝓝 x] 1


instance : FunLike (PartitionOfUnity ι X s) ι C(X, ℝ) where
  coe := toFun
                             /-
                               ι : Type u
                               X : Type v
                               inst✝⁴ : TopologicalSpace X
                               E : Type u_1
                               inst✝³ : AddCommMonoid E
                               inst✝² : SMulWithZero Real E
                               inst✝¹ : TopologicalSpace E
                               inst✝ : ContinuousSMul Real E
                               s : Set X
                               f✝ f g : PartitionOfUnity ι X s
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


protected theorem locallyFinite : LocallyFinite fun i => support (f i) :=
  f.locallyFinite'


theorem locallyFinite_tsupport : LocallyFinite fun i => tsupport (f i) :=
  f.locallyFinite.closure


theorem nonneg (i : ι) (x : X) : 0 ≤ f i x :=
  f.nonneg' i x


theorem sum_eq_one {x : X} (hx : x ∈ s) : ∑ᶠ i, f i x = 1 :=
  f.sum_eq_one' x hx


/-- If `f` is a partition of unity on `s`, then for every `x ∈ s` there exists an index `i` such
that `0 < f i x`. -/
theorem exists_pos {x : X} (hx : x ∈ s) : ∃ i, 0 < f i x := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : PartitionOfUnity ι X s
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun i => LT.lt 0 ((f i) x)
  -/
  have H := f.sum_eq_one hx
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : PartitionOfUnity ι X s
    x : X
    hx : Membership.mem s x
    H : Eq (finsum fun i => (f i) x) 1
    ⊢ Exists fun i => LT.lt 0 ((f i) x)
  -/
  contrapose! H
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : PartitionOfUnity ι X s
    x : X
    hx : Membership.mem s x
    H : ∀ (i : ι), LE.le ((f i) x) 0
    ⊢ Ne (finsum fun i => (f i) x) 1
  -/
  simpa only [fun i => (H i).antisymm (f.nonneg i x), finsum_zero] using zero_ne_one
  /-
    🎉 no goals
  -/


theorem sum_le_one (x : X) : ∑ᶠ i, f i x ≤ 1 :=
  f.sum_le_one' x


theorem sum_nonneg (x : X) : 0 ≤ ∑ᶠ i, f i x :=
  finsum_nonneg fun i => f.nonneg i x


theorem le_one (i : ι) (x : X) : f i x ≤ 1 :=
  (single_le_finsum i (f.locallyFinite.point_finite x) fun j => f.nonneg j x).trans (f.sum_le_one x)


/-- The support of a partition of unity at a point `x₀` as a `Finset`.
  This is the set of `i : ι` such that `x₀ ∈ support f i`, i.e. `f i ≠ x₀`. -/
def finsupport : Finset ι := (ρ.locallyFinite.point_finite x₀).toFinset


@[simp]
theorem mem_finsupport (x₀ : X) {i} :
    i ∈ ρ.finsupport x₀ ↔ i ∈ support fun i ↦ ρ i x₀ := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    i : ι
    ⊢ Iff (Membership.mem (ρ.finsupport x₀) i) (Membership.mem (Function.support f …
  -/
  simp only [finsupport, mem_support, Finite.mem_toFinset, mem_setOf_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_finsupport (x₀ : X) :
    (ρ.finsupport x₀ : Set ι) = support fun i ↦ ρ i x₀ := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    ⊢ Eq (↑(ρ.finsupport x₀)) (Function.support fun i => (ρ i) x₀)
  -/
  ext
  /-
    case h
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    x✝ : ι
    ⊢ Iff (Membership.mem (↑(ρ.finsupport x₀)) x✝) (Membership.mem (Function.suppo …
  -/
  rw [Finset.mem_coe, mem_finsupport]
  /-
    🎉 no goals
  -/


theorem sum_finsupport (hx₀ : x₀ ∈ s) : ∑ i ∈ ρ.finsupport x₀, ρ i x₀ = 1 := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    hx₀ : Membership.mem s x₀
    ⊢ Eq ((ρ.finsupport x₀).sum fun i => (ρ i) x₀) 1
  -/
  rw [← ρ.sum_eq_one hx₀, finsum_eq_sum_of_support_subset _ (ρ.coe_finsupport x₀).superset]
  /-
    🎉 no goals
  -/


theorem sum_finsupport' (hx₀ : x₀ ∈ s) {I : Finset ι} (hI : ρ.finsupport x₀ ⊆ I) :
    ∑ i ∈ I, ρ i x₀ = 1 := by
  classical
  rw [← Finset.sum_sdiff hI, ρ.sum_finsupport hx₀]
  suffices ∑ i ∈ I \ ρ.finsupport x₀, (ρ i) x₀ = ∑ i ∈ I \ ρ.finsupport x₀, 0 by
    rw [this, add_left_eq_self, Finset.sum_const_zero]
  apply Finset.sum_congr rfl
  rintro x hx
  simp only [Finset.mem_sdiff, ρ.mem_finsupport, mem_support, Classical.not_not] at hx
  exact hx.2


theorem sum_finsupport_smul_eq_finsum {M : Type*} [AddCommGroup M] [Module ℝ M] (φ : ι → X → M) :
    ∑ i ∈ ρ.finsupport x₀, ρ i x₀ • φ i x₀ = ∑ᶠ i, ρ i x₀ • φ i x₀ := by
  /-
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module Real M
    φ : ι → X → M
    ⊢ Eq ((ρ.finsupport x₀).sum fun i => HSMul.hSMul ((ρ i) x₀) (φ i x₀)) (finsum  …
  -/
  apply (finsum_eq_sum_of_support_subset _ _).symm
  have : (fun i ↦ (ρ i) x₀ • φ i x₀) = (fun i ↦ (ρ i) x₀) • (fun i ↦ φ i x₀) :=
    funext fun _ => (Pi.smul_apply' _ _ _).symm
  /-
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module Real M
    φ : ι → X → M
    this : Eq (fun i => HSMul.hSMul ((ρ i) x₀) (φ i x₀)) (HSMul.hSMul (fun i => (ρ …
    ⊢ HasSubset.Subset (Function.support fun i => HSMul.hSMul ((ρ i) x₀) (φ i x₀)) …
  -/
  rw [ρ.coe_finsupport x₀, this, support_smul]
  /-
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module Real M
    φ : ι → X → M
    this : Eq (fun i => HSMul.hSMul ((ρ i) x₀) (φ i x₀)) (HSMul.hSMul (fun i => (ρ …
    ⊢ HasSubset.Subset (Inter.inter (Function.support fun i => (ρ i) x₀) (Function …
  -/
  exact inter_subset_left
  /-
    🎉 no goals
  -/


/-- The `tsupport`s of a partition of unity are locally finite. -/
theorem finite_tsupport : {i | x₀ ∈ tsupport (ρ i)}.Finite := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    ⊢ (setOf fun i => Membership.mem (tsupport ⇑(ρ i)) x₀).Finite
  -/
  rcases ρ.locallyFinite x₀ with ⟨t, t_in, ht⟩
  /-
    case intro.intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    t : Set X
    t_in : Membership.mem (nhds x₀) t
    ht : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(ρ i)) i) t).No …
    ⊢ (setOf fun i => Membership.mem (tsupport ⇑(ρ i)) x₀).Finite
  -/
  apply ht.subset
  /-
    case intro.intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    t : Set X
    t_in : Membership.mem (nhds x₀) t
    ht : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(ρ i)) i) t).No …
    ⊢ HasSubset.Subset (setOf fun i => Membership.mem (tsupport ⇑(ρ i)) x₀) (setOf …
  -/
  rintro i hi
  /-
    case intro.intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    t : Set X
    t_in : Membership.mem (nhds x₀) t
    ht : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(ρ i)) i) t).No …
    i : ι
    hi : Membership.mem (setOf fun i => Membership.mem (tsupport ⇑(ρ i)) x₀) i
    ⊢ Membership.mem (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(ρ  …
  -/
  simp only [inter_comm]
  /-
    case intro.intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    t : Set X
    t_in : Membership.mem (nhds x₀) t
    ht : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(ρ i)) i) t).No …
    i : ι
    hi : Membership.mem (setOf fun i => Membership.mem (tsupport ⇑(ρ i)) x₀) i
    ⊢ Membership.mem (setOf fun i => (Inter.inter t (Function.support ⇑(ρ i))).Non …
  -/
  exact mem_closure_iff_nhds.mp hi t t_in
  /-
    🎉 no goals
  -/


/-- The tsupport of a partition of unity at a point `x₀` as a `Finset`.
  This is the set of `i : ι` such that `x₀ ∈ tsupport f i`. -/
def fintsupport (x₀ : X) : Finset ι :=
  (ρ.finite_tsupport x₀).toFinset


theorem mem_fintsupport_iff (i : ι) : i ∈ ρ.fintsupport x₀ ↔ x₀ ∈ tsupport (ρ i) :=
  Finite.mem_toFinset _


theorem eventually_fintsupport_subset :
    ∀ᶠ y in 𝓝 x₀, ρ.fintsupport y ⊆ ρ.fintsupport x₀ := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    ⊢ Filter.Eventually (fun y => HasSubset.Subset (ρ.fintsupport y) (ρ.fintsuppor …
  -/
  apply (ρ.locallyFinite.closure.eventually_subset (fun _ ↦ isClosed_closure) x₀).mono
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    ⊢ ∀ (x : X), HasSubset.Subset (setOf fun i => Membership.mem (closure (Functio …
  -/
  intro y hy z hz
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ y : X
    hy : HasSubset.Subset (setOf fun i => Membership.mem (closure (Function.suppor …
    z : ι
    hz : Membership.mem (ρ.fintsupport y) z
    ⊢ Membership.mem (ρ.fintsupport x₀) z
  -/
  rw [PartitionOfUnity.mem_fintsupport_iff] at *
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ y : X
    hy : HasSubset.Subset (setOf fun i => Membership.mem (closure (Function.suppor …
    z : ι
    hz : Membership.mem (tsupport ⇑(ρ z)) y
    ⊢ Membership.mem (tsupport ⇑(ρ z)) x₀
  -/
  exact hy hz
  /-
    🎉 no goals
  -/


theorem finsupport_subset_fintsupport : ρ.finsupport x₀ ⊆ ρ.fintsupport x₀ := fun i hi ↦ by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    i : ι
    hi : Membership.mem (ρ.finsupport x₀) i
    ⊢ Membership.mem (ρ.fintsupport x₀) i
  -/
  rw [ρ.mem_fintsupport_iff]
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    i : ι
    hi : Membership.mem (ρ.finsupport x₀) i
    ⊢ Membership.mem (tsupport ⇑(ρ i)) x₀
  -/
  apply subset_closure
  /-
    case a
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    i : ι
    hi : Membership.mem (ρ.finsupport x₀) i
    ⊢ Membership.mem (Function.support ⇑(ρ i)) x₀
  -/
  exact (ρ.mem_finsupport x₀).mp hi
  /-
    🎉 no goals
  -/


theorem eventually_finsupport_subset : ∀ᶠ y in 𝓝 x₀, ρ.finsupport y ⊆ ρ.fintsupport x₀ :=
  (ρ.eventually_fintsupport_subset x₀).mono
    fun y hy ↦ (ρ.finsupport_subset_fintsupport y).trans hy


/-- If `f` is a partition of unity on `s : Set X` and `g : X → E` is continuous at every point of
the topological support of some `f i`, then `fun x ↦ f i x • g x` is continuous on the whole space.
-/
theorem continuous_smul {g : X → E} {i : ι} (hg : ∀ x ∈ tsupport (f i), ContinuousAt g x) :
    Continuous fun x => f i x • g x :=
  continuous_of_tsupport fun x hx =>
    ((f i).continuousAt x).smul <| hg x <| tsupport_smul_subset_left _ _ hx


/-- If `f` is a partition of unity on a set `s : Set X` and `g : ι → X → E` is a family of functions
such that each `g i` is continuous at every point of the topological support of `f i`, then the sum
`fun x ↦ ∑ᶠ i, f i x • g i x` is continuous on the whole space. -/
theorem continuous_finsum_smul [ContinuousAdd E] {g : ι → X → E}
    (hg : ∀ (i), ∀ x ∈ tsupport (f i), ContinuousAt (g i) x) :
    Continuous fun x => ∑ᶠ i, f i x • g i x :=
  (continuous_finsum fun i => f.continuous_smul (hg i)) <|
    f.locallyFinite.subset fun _ => support_smul_subset_left _ _


/-- A partition of unity `f i` is subordinate to a family of sets `U i` indexed by the same type if
for each `i` the closure of the support of `f i` is a subset of `U i`. -/
def IsSubordinate (U : ι → Set X) : Prop :=
  ∀ i, tsupport (f i) ⊆ U i


theorem exists_finset_nhd' {s : Set X} (ρ : PartitionOfUnity ι X s) (x₀ : X) :
    ∃ I : Finset ι, (∀ᶠ x in 𝓝[s] x₀, ∑ i ∈ I, ρ i x = 1) ∧
      ∀ᶠ x in 𝓝 x₀, support (ρ · x) ⊆ I := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    ⊢ Exists fun I => And (Filter.Eventually (fun x => Eq (I.sum fun i => (ρ i) x) …
  -/
  rcases ρ.locallyFinite.exists_finset_support x₀ with ⟨I, hI⟩
  /-
    case intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    I : Finset ι
    hI : Filter.Eventually (fun x => HasSubset.Subset (Function.support fun i => ( …
    ⊢ Exists fun I => And (Filter.Eventually (fun x => Eq (I.sum fun i => (ρ i) x) …
  -/
  refine ⟨I, eventually_nhdsWithin_iff.mpr (hI.mono fun x hx x_in ↦ ?_), hI⟩
  /-
    case intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    I : Finset ι
    hI : Filter.Eventually (fun x => HasSubset.Subset (Function.support fun i => ( …
    x : X
    hx : HasSubset.Subset (Function.support fun i => (ρ i) x) ↑I
    x_in : Membership.mem s x
    ⊢ Eq (I.sum fun i => (ρ i) x) 1
  -/
  have : ∑ᶠ i : ι, ρ i x = ∑ i ∈ I, ρ i x := finsum_eq_sum_of_support_subset _ hx
  /-
    case intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    ρ : PartitionOfUnity ι X s
    x₀ : X
    I : Finset ι
    hI : Filter.Eventually (fun x => HasSubset.Subset (Function.support fun i => ( …
    x : X
    hx : HasSubset.Subset (Function.support fun i => (ρ i) x) ↑I
    x_in : Membership.mem s x
    this : Eq (finsum fun i => (ρ i) x) (I.sum fun i => (ρ i) x)
    ⊢ Eq (I.sum fun i => (ρ i) x) 1
  -/
  rwa [eq_comm, ρ.sum_eq_one x_in] at this
  /-
    🎉 no goals
  -/


theorem exists_finset_nhd (ρ : PartitionOfUnity ι X univ) (x₀ : X) :
    ∃ I : Finset ι, ∀ᶠ x in 𝓝 x₀, ∑ i ∈ I, ρ i x = 1 ∧ support (ρ · x) ⊆ I := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    ρ : PartitionOfUnity ι X
    x₀ : X
    ⊢ Exists fun I => Filter.Eventually (fun x => And (Eq (I.sum fun i => (ρ i) x) …
  -/
  rcases ρ.exists_finset_nhd' x₀ with ⟨I, H⟩
  /-
    case intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    ρ : PartitionOfUnity ι X
    x₀ : X
    I : Finset ι
    H : And (Filter.Eventually (fun x => Eq (I.sum fun i => (ρ i) x) 1) (nhdsWithi …
    ⊢ Exists fun I => Filter.Eventually (fun x => And (Eq (I.sum fun i => (ρ i) x) …
  -/
  use I
  /-
    case h
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    ρ : PartitionOfUnity ι X
    x₀ : X
    I : Finset ι
    H : And (Filter.Eventually (fun x => Eq (I.sum fun i => (ρ i) x) 1) (nhdsWithi …
    ⊢ Filter.Eventually (fun x => And (Eq (I.sum fun i => (ρ i) x) 1) (HasSubset.S …
  -/
  rwa [nhdsWithin_univ, ← eventually_and] at H
  /-
    🎉 no goals
  -/


theorem exists_finset_nhd_support_subset {U : ι → Set X} (hso : f.IsSubordinate U)
    (ho : ∀ i, IsOpen (U i)) (x : X) :
    ∃ is : Finset ι, ∃ n ∈ 𝓝 x, n ⊆ ⋂ i ∈ is, U i ∧ ∀ z ∈ n, (support (f · z)) ⊆ is :=
  f.locallyFinite.exists_finset_nhd_support_subset hso ho x


/-- If `f` is a partition of unity that is subordinate to a family of open sets `U i` and
`g : ι → X → E` is a family of functions such that each `g i` is continuous on `U i`, then the sum
`fun x ↦ ∑ᶠ i, f i x • g i x` is a continuous function. -/
theorem IsSubordinate.continuous_finsum_smul [ContinuousAdd E] {U : ι → Set X}
    (ho : ∀ i, IsOpen (U i)) (hf : f.IsSubordinate U) {g : ι → X → E}
    (hg : ∀ i, ContinuousOn (g i) (U i)) : Continuous fun x => ∑ᶠ i, f i x • g i x :=
  f.continuous_finsum_smul fun i _ hx => (hg i).continuousAt <| (ho i).mem_nhds <| hf i hx


instance : FunLike (BumpCovering ι X s) ι C(X, ℝ) where
  coe := toFun
                             /-
                               ι : Type u
                               X : Type v
                               inst✝ : TopologicalSpace X
                               s : Set X
                               f✝ f g : BumpCovering ι X s
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[simp] lemma toFun_eq_coe : f.toFun = f := rfl


protected theorem point_finite (x : X) : { i | f i x ≠ 0 }.Finite :=
  f.locallyFinite.point_finite x


theorem le_one (i : ι) (x : X) : f i x ≤ 1 :=
  f.le_one' i x


open Classical in
/-- A `BumpCovering` that consists of a single function, uniformly equal to one, defined as an
example for `Inhabited` instance. -/
protected def single (i : ι) (s : Set X) : BumpCovering ι X s where
  toFun := Pi.single i 1
  locallyFinite' x := by
    /-
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s✝ : Set X
      f : BumpCovering ι X s✝
      i : ι
      s : Set X
      x : X
      ⊢ Exists fun t => And (Membership.mem (nhds x) t) (setOf fun i_1 => (Inter.int …
    -/
    refine ⟨univ, univ_mem, (finite_singleton i).subset ?_⟩
    /-
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s✝ : Set X
      f : BumpCovering ι X s✝
      i : ι
      s : Set X
      x : X
      ⊢ HasSubset.Subset (setOf fun i_1 => (Inter.inter ((fun i_2 => Function.suppor …
    -/
    rintro j ⟨x, hx, -⟩
    /-
      case intro.intro
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s✝ : Set X
      f : BumpCovering ι X s✝
      i : ι
      s : Set X
      x✝ : X
      j : ι
      x : X
      hx : Membership.mem ((fun i_1 => Function.support ⇑(Pi.single i 1 i_1)) j) x
      ⊢ Membership.mem (Singleton.singleton i) j
    -/
    contrapose! hx
    /-
      case intro.intro
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s✝ : Set X
      f : BumpCovering ι X s✝
      i : ι
      s : Set X
      x✝ : X
      j : ι
      x : X
      hx : Not (Membership.mem (Singleton.singleton i) j)
      ⊢ Not (Membership.mem (Function.support ⇑(Pi.single i 1 j)) x)
    -/
    rw [mem_singleton_iff] at hx
    /-
      case intro.intro
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s✝ : Set X
      f : BumpCovering ι X s✝
      i : ι
      s : Set X
      x✝ : X
      j : ι
      x : X
      hx : Not (Eq j i)
      ⊢ Not (Membership.mem (Function.support ⇑(Pi.single i 1 j)) x)
    -/
    simp [hx]
    /-
      🎉 no goals
    -/
  nonneg' := le_update_iff.2 ⟨fun _ => zero_le_one, fun _ _ => le_rfl⟩
  le_one' := update_le_iff.2 ⟨le_rfl, fun _ _ _ => zero_le_one⟩
                                  /-
                                    ι : Type u
                                    X : Type v
                                    inst✝ : TopologicalSpace X
                                    s✝ : Set X
                                    f : BumpCovering ι X s✝
                                    i : ι
                                    s : Set X
                                    x : X
                                    x✝ : Membership.mem s x
                                    ⊢ (nhds x).EventuallyEq (⇑(Pi.single i 1 i)) 1
                                  -/
  eventuallyEq_one' x _ := ⟨i, by rw [Pi.single_eq_same, ContinuousMap.coe_one]⟩
                                  /-
                                    🎉 no goals
                                  -/


open Classical in
@[simp]
theorem coe_single (i : ι) (s : Set X) : ⇑(BumpCovering.single i s) = Pi.single i 1 := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    i : ι
    s : Set X
    ⊢ Eq (⇑(BumpCovering.single i s)) (Pi.single i 1)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance [Inhabited ι] : Inhabited (BumpCovering ι X s) :=
  ⟨BumpCovering.single default s⟩


/-- A collection of bump functions `f i` is subordinate to a family of sets `U i` indexed by the
same type if for each `i` the closure of the support of `f i` is a subset of `U i`. -/
def IsSubordinate (f : BumpCovering ι X s) (U : ι → Set X) : Prop :=
  ∀ i, tsupport (f i) ⊆ U i


theorem IsSubordinate.mono {f : BumpCovering ι X s} {U V : ι → Set X} (hU : f.IsSubordinate U)
    (hV : ∀ i, U i ⊆ V i) : f.IsSubordinate V :=
  fun i => Subset.trans (hU i) (hV i)


/-- If `X` is a normal topological space and `U i`, `i : ι`, is a locally finite open covering of a
closed set `s`, then there exists a `BumpCovering ι X s` that is subordinate to `U`. If `X` is a
paracompact space, then the assumption `hf : LocallyFinite U` can be omitted, see
`BumpCovering.exists_isSubordinate`. This version assumes that `p : (X → ℝ) → Prop` is a predicate
that satisfies Urysohn's lemma, and provides a `BumpCovering` such that each function of the
covering satisfies `p`. -/
theorem exists_isSubordinate_of_locallyFinite_of_prop [NormalSpace X] (p : (X → ℝ) → Prop)
    (h01 : ∀ s t, IsClosed s → IsClosed t → Disjoint s t →
      ∃ f : C(X, ℝ), p f ∧ EqOn f 0 s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1)
    (hs : IsClosed s) (U : ι → Set X) (ho : ∀ i, IsOpen (U i)) (hf : LocallyFinite U)
    (hU : s ⊆ ⋃ i, U i) : ∃ f : BumpCovering ι X s, (∀ i, p (f i)) ∧ f.IsSubordinate U := by
  rcases exists_subset_iUnion_closure_subset hs ho (fun x _ => hf.point_finite x) hU with
    ⟨V, hsV, hVo, hVU⟩
  /-
    case intro.intro.intro
    ι : Type u
    X : Type v
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : NormalSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (f.IsSubordinate U)
  -/
  have hVU' : ∀ i, V i ⊆ U i := fun i => Subset.trans subset_closure (hVU i)
  rcases exists_subset_iUnion_closure_subset hs hVo (fun x _ => (hf.subset hVU').point_finite x)
      hsV with
    ⟨W, hsW, hWo, hWV⟩
  choose f hfp hf0 hf1 hf01 using fun i =>
    h01 _ _ (isClosed_compl_iff.2 <| hVo i) isClosed_closure
      (disjoint_right.2 fun x hx => Classical.not_not.2 (hWV i hx))
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : NormalSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hVU' : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    W : ι → Set X
    hsW : HasSubset.Subset s (Set.iUnion W)
    hWo : ∀ (i : ι), IsOpen (W i)
    hWV : ∀ (i : ι), HasSubset.Subset (closure (W i)) (V i)
    f : ι → ContinuousMap X Real
    hfp : ∀ (i : ι), p ⇑(f i)
    hf0 : ∀ (i : ι), Set.EqOn (⇑(f i)) 0 (HasCompl.compl (V i))
    hf1 : ∀ (i : ι), Set.EqOn (⇑(f i)) 1 (closure (W i))
    hf01 : ∀ (i : ι) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (f.IsSubordinate U)
  -/
  have hsupp : ∀ i, support (f i) ⊆ V i := fun i => support_subset_iff'.2 (hf0 i)
  refine ⟨⟨f, hf.subset fun i => Subset.trans (hsupp i) (hVU' i), fun i x => (hf01 i x).1,
      fun i x => (hf01 i x).2, fun x hx => ?_⟩,
    hfp, fun i => Subset.trans (closure_mono (hsupp i)) (hVU i)⟩
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : NormalSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hVU' : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    W : ι → Set X
    hsW : HasSubset.Subset s (Set.iUnion W)
    hWo : ∀ (i : ι), IsOpen (W i)
    hWV : ∀ (i : ι), HasSubset.Subset (closure (W i)) (V i)
    f : ι → ContinuousMap X Real
    hfp : ∀ (i : ι), p ⇑(f i)
    hf0 : ∀ (i : ι), Set.EqOn (⇑(f i)) 0 (HasCompl.compl (V i))
    hf1 : ∀ (i : ι), Set.EqOn (⇑(f i)) 1 (closure (W i))
    hf01 : ∀ (i : ι) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
    hsupp : ∀ (i : ι), HasSubset.Subset (Function.support ⇑(f i)) (V i)
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun i => (nhds x).EventuallyEq (⇑(f i)) 1
  -/
  rcases mem_iUnion.1 (hsW hx) with ⟨i, hi⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝¹ : TopologicalSpace X
    s : Set X
    inst✝ : NormalSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hVU' : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    W : ι → Set X
    hsW : HasSubset.Subset s (Set.iUnion W)
    hWo : ∀ (i : ι), IsOpen (W i)
    hWV : ∀ (i : ι), HasSubset.Subset (closure (W i)) (V i)
    f : ι → ContinuousMap X Real
    hfp : ∀ (i : ι), p ⇑(f i)
    hf0 : ∀ (i : ι), Set.EqOn (⇑(f i)) 0 (HasCompl.compl (V i))
    hf1 : ∀ (i : ι), Set.EqOn (⇑(f i)) 1 (closure (W i))
    hf01 : ∀ (i : ι) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
    hsupp : ∀ (i : ι), HasSubset.Subset (Function.support ⇑(f i)) (V i)
    x : X
    hx : Membership.mem s x
    i : ι
    hi : Membership.mem (W i) x
    ⊢ Exists fun i => (nhds x).EventuallyEq (⇑(f i)) 1
  -/
  exact ⟨i, ((hf1 i).mono subset_closure).eventuallyEq_of_mem ((hWo i).mem_nhds hi)⟩
  /-
    🎉 no goals
  -/


/-- If `X` is a normal topological space and `U i`, `i : ι`, is a locally finite open covering of a
closed set `s`, then there exists a `BumpCovering ι X s` that is subordinate to `U`. If `X` is a
paracompact space, then the assumption `hf : LocallyFinite U` can be omitted, see
`BumpCovering.exists_isSubordinate`. -/
theorem exists_isSubordinate_of_locallyFinite [NormalSpace X] (hs : IsClosed s) (U : ι → Set X)
    (ho : ∀ i, IsOpen (U i)) (hf : LocallyFinite U) (hU : s ⊆ ⋃ i, U i) :
    ∃ f : BumpCovering ι X s, f.IsSubordinate U :=
  let ⟨f, _, hfU⟩ :=
    exists_isSubordinate_of_locallyFinite_of_prop (fun _ => True)
      (fun _ _ hs ht hd =>
        (exists_continuous_zero_one_of_isClosed hs ht hd).imp fun _ hf => ⟨trivial, hf⟩)
      hs U ho hf hU
  ⟨f, hfU⟩


/-- If `X` is a paracompact normal topological space and `U` is an open covering of a closed set
`s`, then there exists a `BumpCovering ι X s` that is subordinate to `U`. This version assumes that
`p : (X → ℝ) → Prop` is a predicate that satisfies Urysohn's lemma, and provides a
`BumpCovering` such that each function of the covering satisfies `p`. -/
theorem exists_isSubordinate_of_prop [NormalSpace X] [ParacompactSpace X] (p : (X → ℝ) → Prop)
    (h01 : ∀ s t, IsClosed s → IsClosed t → Disjoint s t →
      ∃ f : C(X, ℝ), p f ∧ EqOn f 0 s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1)
    (hs : IsClosed s) (U : ι → Set X) (ho : ∀ i, IsOpen (U i)) (hU : s ⊆ ⋃ i, U i) :
    ∃ f : BumpCovering ι X s, (∀ i, p (f i)) ∧ f.IsSubordinate U := by
  /-
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : NormalSpace X
    inst✝ : ParacompactSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (f.IsSubordinate U)
  -/
  rcases precise_refinement_set hs _ ho hU with ⟨V, hVo, hsV, hVf, hVU⟩
  /-
    case intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : NormalSpace X
    inst✝ : ParacompactSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hVo : ∀ (i : ι), IsOpen (V i)
    hsV : HasSubset.Subset s (Set.iUnion fun i => V i)
    hVf : LocallyFinite V
    hVU : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (f.IsSubordinate U)
  -/
  rcases exists_isSubordinate_of_locallyFinite_of_prop p h01 hs V hVo hVf hsV with ⟨f, hfp, hf⟩
  /-
    case intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : NormalSpace X
    inst✝ : ParacompactSpace X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsClosed t → Disjoint s t → Exists fun f = …
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hVo : ∀ (i : ι), IsOpen (V i)
    hsV : HasSubset.Subset s (Set.iUnion fun i => V i)
    hVf : LocallyFinite V
    hVU : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    f : BumpCovering ι X s
    hfp : ∀ (i : ι), p ⇑(f i)
    hf : f.IsSubordinate V
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (f.IsSubordinate U)
  -/
  exact ⟨f, hfp, hf.mono hVU⟩
  /-
    🎉 no goals
  -/


/-- If `X` is a paracompact normal topological space and `U` is an open covering of a closed set
`s`, then there exists a `BumpCovering ι X s` that is subordinate to `U`. -/
theorem exists_isSubordinate [NormalSpace X] [ParacompactSpace X] (hs : IsClosed s) (U : ι → Set X)
    (ho : ∀ i, IsOpen (U i)) (hU : s ⊆ ⋃ i, U i) : ∃ f : BumpCovering ι X s, f.IsSubordinate U := by
  /-
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : NormalSpace X
    inst✝ : ParacompactSpace X
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    ⊢ Exists fun f => f.IsSubordinate U
  -/
  rcases precise_refinement_set hs _ ho hU with ⟨V, hVo, hsV, hVf, hVU⟩
  /-
    case intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : NormalSpace X
    inst✝ : ParacompactSpace X
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hVo : ∀ (i : ι), IsOpen (V i)
    hsV : HasSubset.Subset s (Set.iUnion fun i => V i)
    hVf : LocallyFinite V
    hVU : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    ⊢ Exists fun f => f.IsSubordinate U
  -/
  rcases exists_isSubordinate_of_locallyFinite hs V hVo hVf hsV with ⟨f, hf⟩
  /-
    case intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : NormalSpace X
    inst✝ : ParacompactSpace X
    hs : IsClosed s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hVo : ∀ (i : ι), IsOpen (V i)
    hsV : HasSubset.Subset s (Set.iUnion fun i => V i)
    hVf : LocallyFinite V
    hVU : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    f : BumpCovering ι X s
    hf : f.IsSubordinate V
    ⊢ Exists fun f => f.IsSubordinate U
  -/
  exact ⟨f, hf.mono hVU⟩
  /-
    🎉 no goals
  -/


/-- If `X` is a locally compact T2 topological space and `U i`, `i : ι`, is a locally finite open
covering of a compact set `s`, then there exists a `BumpCovering ι X s` that is subordinate to `U`.
If `X` is a paracompact space, then the assumption `hf : LocallyFinite U` can be omitted, see
`BumpCovering.exists_isSubordinate`. This version assumes that `p : (X → ℝ) → Prop` is a predicate
that satisfies Urysohn's lemma, and provides a `BumpCovering` such that each function of the
covering satisfies `p`. -/
theorem exists_isSubordinate_of_locallyFinite_of_prop_t2space [LocallyCompactSpace X] [T2Space X]
    (p : (X → ℝ) → Prop) (h01 : ∀ s t, IsClosed s → IsCompact t → Disjoint s t → ∃ f : C(X, ℝ),
    p f ∧ EqOn f 0 s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1)
    (hs : IsCompact s) (U : ι → Set X) (ho : ∀ i, IsOpen (U i)) (hf : LocallyFinite U)
    (hU : s ⊆ ⋃ i, U i) :
    ∃ f : BumpCovering ι X s, (∀ i, p (f i)) ∧ f.IsSubordinate U ∧
      ∀ i, HasCompactSupport (f i) := by
  rcases exists_subset_iUnion_closure_subset_t2space hs ho (fun x _ => hf.point_finite x) hU with
    ⟨V, hsV, hVo, hVU, hcp⟩
  /-
    case intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : T2Space X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsCompact t → Disjoint s t → Exists fun f  …
    hs : IsCompact s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hcp : ∀ (i : ι), IsCompact (closure (V i))
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (And (f.IsSubordinate U) (∀ (i : ι …
  -/
  have hVU' i : V i ⊆ U i := subset_closure.trans (hVU i)
  rcases exists_subset_iUnion_closure_subset_t2space hs hVo
    (fun x _ => (hf.subset hVU').point_finite x) hsV with ⟨W, hsW, hWo, hWV, hWc⟩
  choose f hfp hf0 hf1 hf01 using fun i =>
    h01 _ _ (isClosed_compl_iff.2 <| hVo i) (hWc i)
      (disjoint_right.2 fun x hx => Classical.not_not.2 (hWV i hx))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : T2Space X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsCompact t → Disjoint s t → Exists fun f  …
    hs : IsCompact s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hcp : ∀ (i : ι), IsCompact (closure (V i))
    hVU' : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    W : ι → Set X
    hsW : HasSubset.Subset s (Set.iUnion W)
    hWo : ∀ (i : ι), IsOpen (W i)
    hWV : ∀ (i : ι), HasSubset.Subset (closure (W i)) (V i)
    hWc : ∀ (i : ι), IsCompact (closure (W i))
    f : ι → ContinuousMap X Real
    hfp : ∀ (i : ι), p ⇑(f i)
    hf0 : ∀ (i : ι), Set.EqOn (⇑(f i)) 0 (HasCompl.compl (V i))
    hf1 : ∀ (i : ι), Set.EqOn (⇑(f i)) 1 (closure (W i))
    hf01 : ∀ (i : ι) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
    ⊢ Exists fun f => And (∀ (i : ι), p ⇑(f i)) (And (f.IsSubordinate U) (∀ (i : ι …
  -/
  have hsupp i : support (f i) ⊆ V i := support_subset_iff'.2 (hf0 i)
  refine ⟨⟨f, hf.subset fun i => Subset.trans (hsupp i) (hVU' i), fun i x => (hf01 i x).1,
      fun i x => (hf01 i x).2, fun x hx => ?_⟩,
    hfp, fun i => Subset.trans (closure_mono (hsupp i)) (hVU i),
    fun i => IsCompact.of_isClosed_subset (hcp i) isClosed_closure <| closure_mono (hsupp i)⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : T2Space X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsCompact t → Disjoint s t → Exists fun f  …
    hs : IsCompact s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hcp : ∀ (i : ι), IsCompact (closure (V i))
    hVU' : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    W : ι → Set X
    hsW : HasSubset.Subset s (Set.iUnion W)
    hWo : ∀ (i : ι), IsOpen (W i)
    hWV : ∀ (i : ι), HasSubset.Subset (closure (W i)) (V i)
    hWc : ∀ (i : ι), IsCompact (closure (W i))
    f : ι → ContinuousMap X Real
    hfp : ∀ (i : ι), p ⇑(f i)
    hf0 : ∀ (i : ι), Set.EqOn (⇑(f i)) 0 (HasCompl.compl (V i))
    hf1 : ∀ (i : ι), Set.EqOn (⇑(f i)) 1 (closure (W i))
    hf01 : ∀ (i : ι) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
    hsupp : ∀ (i : ι), HasSubset.Subset (Function.support ⇑(f i)) (V i)
    x : X
    hx : Membership.mem s x
    ⊢ Exists fun i => (nhds x).EventuallyEq (⇑(f i)) 1
  -/
  rcases mem_iUnion.1 (hsW hx) with ⟨i, hi⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    ι : Type u
    X : Type v
    inst✝² : TopologicalSpace X
    s : Set X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : T2Space X
    p : (X → Real) → Prop
    h01 : ∀ (s t : Set X), IsClosed s → IsCompact t → Disjoint s t → Exists fun f  …
    hs : IsCompact s
    U : ι → Set X
    ho : ∀ (i : ι), IsOpen (U i)
    hf : LocallyFinite U
    hU : HasSubset.Subset s (Set.iUnion fun i => U i)
    V : ι → Set X
    hsV : HasSubset.Subset s (Set.iUnion V)
    hVo : ∀ (i : ι), IsOpen (V i)
    hVU : ∀ (i : ι), HasSubset.Subset (closure (V i)) (U i)
    hcp : ∀ (i : ι), IsCompact (closure (V i))
    hVU' : ∀ (i : ι), HasSubset.Subset (V i) (U i)
    W : ι → Set X
    hsW : HasSubset.Subset s (Set.iUnion W)
    hWo : ∀ (i : ι), IsOpen (W i)
    hWV : ∀ (i : ι), HasSubset.Subset (closure (W i)) (V i)
    hWc : ∀ (i : ι), IsCompact (closure (W i))
    f : ι → ContinuousMap X Real
    hfp : ∀ (i : ι), p ⇑(f i)
    hf0 : ∀ (i : ι), Set.EqOn (⇑(f i)) 0 (HasCompl.compl (V i))
    hf1 : ∀ (i : ι), Set.EqOn (⇑(f i)) 1 (closure (W i))
    hf01 : ∀ (i : ι) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
    hsupp : ∀ (i : ι), HasSubset.Subset (Function.support ⇑(f i)) (V i)
    x : X
    hx : Membership.mem s x
    i : ι
    hi : Membership.mem (W i) x
    ⊢ Exists fun i => (nhds x).EventuallyEq (⇑(f i)) 1
  -/
  exact ⟨i, ((hf1 i).mono subset_closure).eventuallyEq_of_mem ((hWo i).mem_nhds hi)⟩
  /-
    🎉 no goals
  -/


/-- If `X` is a normal topological space and `U i`, `i : ι`, is a locally finite open covering of a
closed set `s`, then there exists a `BumpCovering ι X s` that is subordinate to `U`. If `X` is a
paracompact space, then the assumption `hf : LocallyFinite U` can be omitted, see
`BumpCovering.exists_isSubordinate`. -/
theorem exists_isSubordinate_hasCompactSupport_of_locallyFinite_t2space [LocallyCompactSpace X]
    [T2Space X]
    (hs : IsCompact s) (U : ι → Set X) (ho : ∀ i, IsOpen (U i)) (hf : LocallyFinite U)
    (hU : s ⊆ ⋃ i, U i) :
    ∃ f : BumpCovering ι X s, f.IsSubordinate U ∧ ∀ i, HasCompactSupport (f i) := by
  -- need to switch 0 and 1 in `exists_continuous_zero_one_of_isCompact`
  simpa using
    exists_isSubordinate_of_locallyFinite_of_prop_t2space (fun _ => True)
      (fun _ _ ht hs hd =>
        (exists_continuous_zero_one_of_isCompact' hs ht hd.symm).imp fun _ hf => ⟨trivial, hf⟩)
      hs U ho hf hU


/-- Index of a bump function such that `fs i =ᶠ[𝓝 x] 1`. -/
def ind (x : X) (hx : x ∈ s) : ι :=
  (f.eventuallyEq_one' x hx).choose


theorem eventuallyEq_one (x : X) (hx : x ∈ s) : f (f.ind x hx) =ᶠ[𝓝 x] 1 :=
  (f.eventuallyEq_one' x hx).choose_spec


theorem ind_apply (x : X) (hx : x ∈ s) : f (f.ind x hx) x = 1 :=
  (f.eventuallyEq_one x hx).eq_of_nhds


/-- Partition of unity defined by a `BumpCovering`. We use this auxiliary definition to prove some
properties of the new family of functions before bundling it into a `PartitionOfUnity`. Do not use
this definition, use `BumpCovering.toPartitionOfUnity` instead.

The partition of unity is given by the formula `g i x = f i x * ∏ᶠ j < i, (1 - f j x)`. In other
words, `g i x = ∏ᶠ j < i, (1 - f j x) - ∏ᶠ j ≤ i, (1 - f j x)`, so
`∑ᶠ i, g i x = 1 - ∏ᶠ j, (1 - f j x)`. If `x ∈ s`, then one of `f j x` equals one, hence the product
of `1 - f j x` vanishes, and `∑ᶠ i, g i x = 1`.

In order to avoid an assumption `LinearOrder ι`, we use `WellOrderingRel` instead of `(<)`. -/
def toPOUFun (i : ι) (x : X) : ℝ :=
  f i x * ∏ᶠ (j) (_ : WellOrderingRel j i), (1 - f j x)


theorem toPOUFun_zero_of_zero {i : ι} {x : X} (h : f i x = 0) : f.toPOUFun i x = 0 := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    h : Eq ((f i) x) 0
    ⊢ Eq (f.toPOUFun i x) 0
  -/
  rw [toPOUFun, h, zero_mul]
  /-
    🎉 no goals
  -/


theorem support_toPOUFun_subset (i : ι) : support (f.toPOUFun i) ⊆ support (f i) :=
  fun _ => mt <| f.toPOUFun_zero_of_zero


open Classical in
theorem toPOUFun_eq_mul_prod (i : ι) (x : X) (t : Finset ι)
    (ht : ∀ j, WellOrderingRel j i → f j x ≠ 0 → j ∈ t) :
    f.toPOUFun i x = f i x * ∏ j ∈ t.filter fun j => WellOrderingRel j i, (1 - f j x) := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    t : Finset ι
    ht : ∀ (j : ι), WellOrderingRel j i → Ne ((f j) x) 0 → Membership.mem t j
    ⊢ Eq (f.toPOUFun i x) (HMul.hMul ((f i) x) ((Finset.filter (fun j => WellOrder …
  -/
  refine congr_arg _ (finprod_cond_eq_prod_of_cond_iff _ fun {j} hj => ?_)
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    t : Finset ι
    ht : ∀ (j : ι), WellOrderingRel j i → Ne ((f j) x) 0 → Membership.mem t j
    j : ι
    hj : Ne (HSub.hSub 1 ((f j) x)) 1
    ⊢ Iff (WellOrderingRel j i) (Membership.mem (Finset.filter (fun j => WellOrder …
  -/
  rw [Ne, sub_eq_self] at hj
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    t : Finset ι
    ht : ∀ (j : ι), WellOrderingRel j i → Ne ((f j) x) 0 → Membership.mem t j
    j : ι
    hj : Not (Eq ((f j) x) 0)
    ⊢ Iff (WellOrderingRel j i) (Membership.mem (Finset.filter (fun j => WellOrder …
  -/
  rw [Finset.mem_filter, Iff.comm, and_iff_right_iff_imp]
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    t : Finset ι
    ht : ∀ (j : ι), WellOrderingRel j i → Ne ((f j) x) 0 → Membership.mem t j
    j : ι
    hj : Not (Eq ((f j) x) 0)
    ⊢ WellOrderingRel j i → Membership.mem t j
  -/
  exact flip (ht j) hj
  /-
    🎉 no goals
  -/


theorem sum_toPOUFun_eq (x : X) : ∑ᶠ i, f.toPOUFun i x = 1 - ∏ᶠ i, (1 - f i x) := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    x : X
    ⊢ Eq (finsum fun i => f.toPOUFun i x) (HSub.hSub 1 (finprod fun i => HSub.hSub …
  -/
  set s := (f.point_finite x).toFinset
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s✝ : Set X
    f : BumpCovering ι X s✝
    x : X
    s : Finset ι := ⋯.toFinset
    ⊢ Eq (finsum fun i => f.toPOUFun i x) (HSub.hSub 1 (finprod fun i => HSub.hSub …
  -/
  have hs : (s : Set ι) = { i | f i x ≠ 0 } := Finite.coe_toFinset _
  have A : (support fun i => toPOUFun f i x) ⊆ s := by
    rw [hs]
    exact fun i hi => f.support_toPOUFun_subset i hi
  have B : (mulSupport fun i => 1 - f i x) ⊆ s := by
    rw [hs, mulSupport_one_sub]
    exact fun i => id
  classical
  letI : LinearOrder ι := linearOrderOfSTO WellOrderingRel
  rw [finsum_eq_sum_of_support_subset _ A, finprod_eq_prod_of_mulSupport_subset _ B,
    Finset.prod_one_sub_ordered, sub_sub_cancel]
  refine Finset.sum_congr rfl fun i _ => ?_
  convert f.toPOUFun_eq_mul_prod _ _ _ fun j _ hj => _
  rwa [Finite.mem_toFinset]


open Classical in
theorem exists_finset_toPOUFun_eventuallyEq (i : ι) (x : X) : ∃ t : Finset ι,
    f.toPOUFun i =ᶠ[𝓝 x] f i * ∏ j ∈ t.filter fun j => WellOrderingRel j i, (1 - f j) := by
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    ⊢ Exists fun t => (nhds x).EventuallyEq (f.toPOUFun i) (HMul.hMul ⇑(f i) ⇑((Fi …
  -/
  rcases f.locallyFinite x with ⟨U, hU, hf⟩
  /-
    case intro.intro
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    hf : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(f i)) i) U).No …
    ⊢ Exists fun t => (nhds x).EventuallyEq (f.toPOUFun i) (HMul.hMul ⇑(f i) ⇑((Fi …
  -/
  use hf.toFinset
  /-
    case h
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    hf : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(f i)) i) U).No …
    ⊢ (nhds x).EventuallyEq (f.toPOUFun i) (HMul.hMul ⇑(f i) ⇑((Finset.filter (fun …
  -/
  filter_upwards [hU] with y hyU
  /-
    case h
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    hf : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(f i)) i) U).No …
    y : X
    hyU : Membership.mem U y
    ⊢ Eq (f.toPOUFun i y) (HMul.hMul (⇑(f i)) (⇑((Finset.filter (fun j => WellOrde …
  -/
  simp only [ContinuousMap.coe_prod, Pi.mul_apply, Finset.prod_apply]
  /-
    case h
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    hf : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(f i)) i) U).No …
    y : X
    hyU : Membership.mem U y
    ⊢ Eq (f.toPOUFun i y) (HMul.hMul ((f i) y) ((Finset.filter (fun j => WellOrder …
  -/
  apply toPOUFun_eq_mul_prod
  /-
    case h.ht
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    hf : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(f i)) i) U).No …
    y : X
    hyU : Membership.mem U y
    ⊢ ∀ (j : ι), WellOrderingRel j i → Ne ((f j) y) 0 → Membership.mem hf.toFinset j
  -/
  intro j _ hj
  /-
    case h.ht
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    x : X
    U : Set X
    hU : Membership.mem (nhds x) U
    hf : (setOf fun i => (Inter.inter ((fun i => Function.support ⇑(f i)) i) U).No …
    y : X
    hyU : Membership.mem U y
    j : ι
    a✝ : WellOrderingRel j i
    hj : Ne ((f j) y) 0
    ⊢ Membership.mem hf.toFinset j
  -/
  exact hf.mem_toFinset.2 ⟨y, ⟨hj, hyU⟩⟩
  /-
    🎉 no goals
  -/


theorem continuous_toPOUFun (i : ι) : Continuous (f.toPOUFun i) := by
  refine (f i).continuous.mul <|
    continuous_finprod_cond (fun j _ => continuous_const.sub (f j).continuous) ?_
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    ⊢ LocallyFinite fun i => Function.mulSupport fun x => HSub.hSub 1 ((f i) x)
  -/
  simp only [mulSupport_one_sub]
  /-
    ι : Type u
    X : Type v
    inst✝ : TopologicalSpace X
    s : Set X
    f : BumpCovering ι X s
    i : ι
    ⊢ LocallyFinite fun i => Function.support ⇑(f i)
  -/
  exact f.locallyFinite
  /-
    🎉 no goals
  -/


/-- The partition of unity defined by a `BumpCovering`.

The partition of unity is given by the formula `g i x = f i x * ∏ᶠ j < i, (1 - f j x)`. In other
words, `g i x = ∏ᶠ j < i, (1 - f j x) - ∏ᶠ j ≤ i, (1 - f j x)`, so
`∑ᶠ i, g i x = 1 - ∏ᶠ j, (1 - f j x)`. If `x ∈ s`, then one of `f j x` equals one, hence the product
of `1 - f j x` vanishes, and `∑ᶠ i, g i x = 1`.

In order to avoid an assumption `LinearOrder ι`, we use `WellOrderingRel` instead of `(<)`. -/
def toPartitionOfUnity : PartitionOfUnity ι X s where
  toFun i := ⟨f.toPOUFun i, f.continuous_toPOUFun i⟩
  locallyFinite' := f.locallyFinite.subset f.support_toPOUFun_subset
  nonneg' i x :=
    mul_nonneg (f.nonneg i x) (finprod_cond_nonneg fun j _ => sub_nonneg.2 <| f.le_one j x)
  sum_eq_one' x hx := by
    /-
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s : Set X
      f : BumpCovering ι X s
      x : X
      hx : Membership.mem s x
      ⊢ Eq (finsum fun i => ((fun i => { toFun := f.toPOUFun i, continuous_toFun :=  …
    -/
    simp only [ContinuousMap.coe_mk, sum_toPOUFun_eq, sub_eq_self]
    /-
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s : Set X
      f : BumpCovering ι X s
      x : X
      hx : Membership.mem s x
      ⊢ Eq (finprod fun i => HSub.hSub 1 ((f i) x)) 0
    -/
    apply finprod_eq_zero (fun i => 1 - f i x) (f.ind x hx)
      /-
        case hx
        ι : Type u
        X : Type v
        inst✝ : TopologicalSpace X
        s : Set X
        f : BumpCovering ι X s
        x : X
        hx : Membership.mem s x
        ⊢ Eq (HSub.hSub 1 ((f (f.ind x hx)) x)) 0
      -/
    · simp only [f.ind_apply x hx, sub_self]
      /-
        🎉 no goals
      -/
      /-
        case hf
        ι : Type u
        X : Type v
        inst✝ : TopologicalSpace X
        s : Set X
        f : BumpCovering ι X s
        x : X
        hx : Membership.mem s x
        ⊢ (Function.mulSupport fun i => HSub.hSub 1 ((f i) x)).Finite
      -/
    · rw [mulSupport_one_sub]
      /-
        case hf
        ι : Type u
        X : Type v
        inst✝ : TopologicalSpace X
        s : Set X
        f : BumpCovering ι X s
        x : X
        hx : Membership.mem s x
        ⊢ (Function.support fun i => (f i) x).Finite
      -/
      exact f.point_finite x
      /-
        🎉 no goals
      -/
  sum_le_one' x := by
    /-
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s : Set X
      f : BumpCovering ι X s
      x : X
      ⊢ LE.le (finsum fun i => ((fun i => { toFun := f.toPOUFun i, continuous_toFun  …
    -/
    simp only [ContinuousMap.coe_mk, sum_toPOUFun_eq, sub_le_self_iff]
    /-
      ι : Type u
      X : Type v
      inst✝ : TopologicalSpace X
      s : Set X
      f : BumpCovering ι X s
      x : X
      ⊢ LE.le 0 (finprod fun i => HSub.hSub 1 ((f i) x))
    -/
    exact finprod_nonneg fun i => sub_nonneg.2 <| f.le_one i x
    /-
      🎉 no goals
    -/


theorem toPartitionOfUnity_apply (i : ι) (x : X) :
    f.toPartitionOfUnity i x = f i x * ∏ᶠ (j) (_ : WellOrderingRel j i), (1 - f j x) := rfl


open Classical in
theorem toPartitionOfUnity_eq_mul_prod (i : ι) (x : X) (t : Finset ι)
    (ht : ∀ j, WellOrderingRel j i → f j x ≠ 0 → j ∈ t) :
    f.toPartitionOfUnity i x = f i x * ∏ j ∈ t.filter fun j => WellOrderingRel j i, (1 - f j x) :=
  f.toPOUFun_eq_mul_prod i x t ht


open Classical in
theorem exists_finset_toPartitionOfUnity_eventuallyEq (i : ι) (x : X) : ∃ t : Finset ι,
    f.toPartitionOfUnity i =ᶠ[𝓝 x] f i * ∏ j ∈ t.filter fun j => WellOrderingRel j i, (1 - f j) :=
  f.exists_finset_toPOUFun_eventuallyEq i x


theorem toPartitionOfUnity_zero_of_zero {i : ι} {x : X} (h : f i x = 0) :
    f.toPartitionOfUnity i x = 0 :=
  f.toPOUFun_zero_of_zero h


theorem support_toPartitionOfUnity_subset (i : ι) :
    support (f.toPartitionOfUnity i) ⊆ support (f i) :=
  f.support_toPOUFun_subset i


theorem sum_toPartitionOfUnity_eq (x : X) :
    ∑ᶠ i, f.toPartitionOfUnity i x = 1 - ∏ᶠ i, (1 - f i x) :=
  f.sum_toPOUFun_eq x


theorem IsSubordinate.toPartitionOfUnity {f : BumpCovering ι X s} {U : ι → Set X}
    (h : f.IsSubordinate U) : f.toPartitionOfUnity.IsSubordinate U :=
  fun i => Subset.trans (closure_mono <| f.support_toPartitionOfUnity_subset i) (h i)


instance [Inhabited ι] : Inhabited (PartitionOfUnity ι X s) :=
  ⟨BumpCovering.toPartitionOfUnity default⟩


/-- If `X` is a normal topological space and `U` is a locally finite open covering of a closed set
`s`, then there exists a `PartitionOfUnity ι X s` that is subordinate to `U`. If `X` is a
paracompact space, then the assumption `hf : LocallyFinite U` can be omitted, see
`BumpCovering.exists_isSubordinate`. -/
theorem exists_isSubordinate_of_locallyFinite [NormalSpace X] (hs : IsClosed s) (U : ι → Set X)
    (ho : ∀ i, IsOpen (U i)) (hf : LocallyFinite U) (hU : s ⊆ ⋃ i, U i) :
    ∃ f : PartitionOfUnity ι X s, f.IsSubordinate U :=
  let ⟨f, hf⟩ := BumpCovering.exists_isSubordinate_of_locallyFinite hs U ho hf hU
  ⟨f.toPartitionOfUnity, hf.toPartitionOfUnity⟩


/-- If `X` is a paracompact normal topological space and `U` is an open covering of a closed set
`s`, then there exists a `PartitionOfUnity ι X s` that is subordinate to `U`. -/
theorem exists_isSubordinate [NormalSpace X] [ParacompactSpace X] (hs : IsClosed s) (U : ι → Set X)
    (ho : ∀ i, IsOpen (U i)) (hU : s ⊆ ⋃ i, U i) :
    ∃ f : PartitionOfUnity ι X s, f.IsSubordinate U :=
  let ⟨f, hf⟩ := BumpCovering.exists_isSubordinate hs U ho hU
  ⟨f.toPartitionOfUnity, hf.toPartitionOfUnity⟩


/-- If `X` is a locally compact T2 topological space and `U` is a locally finite open covering of a
compact set `s`, then there exists a `PartitionOfUnity ι X s` that is subordinate to `U`. -/
theorem exists_isSubordinate_of_locallyFinite_t2space [LocallyCompactSpace X] [T2Space X]
    (hs : IsCompact s) (U : ι → Set X) (ho : ∀ i, IsOpen (U i)) (hf : LocallyFinite U)
    (hU : s ⊆ ⋃ i, U i) :
    ∃ f : PartitionOfUnity ι X s, f.IsSubordinate U ∧ ∀ i, HasCompactSupport (f i) :=
  let ⟨f, hfsub, hfcp⟩ :=
    BumpCovering.exists_isSubordinate_hasCompactSupport_of_locallyFinite_t2space hs U ho hf hU
  ⟨f.toPartitionOfUnity, hfsub.toPartitionOfUnity, fun i => IsCompact.of_isClosed_subset (hfcp i)
    isClosed_closure <| closure_mono (f.support_toPartitionOfUnity_subset i)⟩


/-- A variation of **Urysohn's lemma**.

In a locally compact T2 space `X`, for a compact set `t` and a finite family of open sets `{s i}_i`
such that `t ⊆ ⋃ i, s i`, there is a family of compactly supported continuous functions `{f i}_i`
supported in `s i`, `∑ i, f i x = 1` on `t` and `0 ≤ f i x ≤ 1`. -/
theorem exists_continuous_sum_one_of_isOpen_isCompact [T2Space X] [LocallyCompactSpace X]
    {n : ℕ} {t : Set X} {s : Fin n → Set X} (hs : ∀ (i : Fin n), IsOpen (s i)) (htcp : IsCompact t)
    (hst : t ⊆ ⋃ i, s i) :
    ∃ f : Fin n → C(X, ℝ), (∀ (i : Fin n), tsupport (f i) ⊆ s i) ∧ EqOn (∑ i, f i) 1 t
      ∧ (∀ (i : Fin n), ∀ (x : X), f i x ∈ Icc (0 : ℝ) 1)
      ∧ (∀ (i : Fin n), HasCompactSupport (f i)) := by
  obtain ⟨f, hfsub, hfcp⟩ := PartitionOfUnity.exists_isSubordinate_of_locallyFinite_t2space htcp s
    hs (locallyFinite_of_finite _) hst
  /-
    case intro.intro
    X : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    n : Nat
    t : Set X
    s : Fin n → Set X
    hs : ∀ (i : Fin n), IsOpen (s i)
    htcp : IsCompact t
    hst : HasSubset.Subset t (Set.iUnion fun i => s i)
    f : PartitionOfUnity (Fin n) X t
    hfsub : f.IsSubordinate s
    hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
    ⊢ Exists fun f => And (∀ (i : Fin n), HasSubset.Subset (tsupport ⇑(f i)) (s i) …
  -/
  use f
  /-
    case h
    X : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    n : Nat
    t : Set X
    s : Fin n → Set X
    hs : ∀ (i : Fin n), IsOpen (s i)
    htcp : IsCompact t
    hst : HasSubset.Subset t (Set.iUnion fun i => s i)
    f : PartitionOfUnity (Fin n) X t
    hfsub : f.IsSubordinate s
    hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
    ⊢ And (∀ (i : Fin n), HasSubset.Subset (tsupport ⇑(f i)) (s i)) (And (Set.EqOn …
  -/
  refine ⟨fun i ↦ hfsub i, ?_, ?_, fun i => hfcp i⟩
    /-
      case h.refine_1
      X : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      n : Nat
      t : Set X
      s : Fin n → Set X
      hs : ∀ (i : Fin n), IsOpen (s i)
      htcp : IsCompact t
      hst : HasSubset.Subset t (Set.iUnion fun i => s i)
      f : PartitionOfUnity (Fin n) X t
      hfsub : f.IsSubordinate s
      hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
      ⊢ Set.EqOn (Finset.univ.sum fun i => ⇑(f i)) 1 t
    -/
  · intro x hx
    /-
      case h.refine_1
      X : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      n : Nat
      t : Set X
      s : Fin n → Set X
      hs : ∀ (i : Fin n), IsOpen (s i)
      htcp : IsCompact t
      hst : HasSubset.Subset t (Set.iUnion fun i => s i)
      f : PartitionOfUnity (Fin n) X t
      hfsub : f.IsSubordinate s
      hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
      x : X
      hx : Membership.mem t x
      ⊢ Eq (Finset.univ.sum (fun i => ⇑(f i)) x) (1 x)
    -/
    simp only [Finset.sum_apply, Pi.one_apply]
    /-
      case h.refine_1
      X : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      n : Nat
      t : Set X
      s : Fin n → Set X
      hs : ∀ (i : Fin n), IsOpen (s i)
      htcp : IsCompact t
      hst : HasSubset.Subset t (Set.iUnion fun i => s i)
      f : PartitionOfUnity (Fin n) X t
      hfsub : f.IsSubordinate s
      hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
      x : X
      hx : Membership.mem t x
      ⊢ Eq (Finset.univ.sum fun c => (f c) x) 1
    -/
    have h := f.sum_eq_one' x hx
    /-
      case h.refine_1
      X : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      n : Nat
      t : Set X
      s : Fin n → Set X
      hs : ∀ (i : Fin n), IsOpen (s i)
      htcp : IsCompact t
      hst : HasSubset.Subset t (Set.iUnion fun i => s i)
      f : PartitionOfUnity (Fin n) X t
      hfsub : f.IsSubordinate s
      hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
      x : X
      hx : Membership.mem t x
      h : Eq (finsum fun i => (f.toFun i) x) 1
      ⊢ Eq (Finset.univ.sum fun c => (f c) x) 1
    -/
    simp at h
    rw [finsum_eq_sum (fun i => (f.toFun i) x)
      (Finite.subset finite_univ (subset_univ (support fun i ↦ (f.toFun i) x)))] at h
    /-
      case h.refine_1
      X : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      n : Nat
      t : Set X
      s : Fin n → Set X
      hs : ∀ (i : Fin n), IsOpen (s i)
      htcp : IsCompact t
      hst : HasSubset.Subset t (Set.iUnion fun i => s i)
      f : PartitionOfUnity (Fin n) X t
      hfsub : f.IsSubordinate s
      hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
      x : X
      hx : Membership.mem t x
      h : Eq (⋯.toFinset.sum fun i => (f.toFun i) x) 1
      ⊢ Eq (Finset.univ.sum fun c => (f c) x) 1
    -/
    simp only [Finite.toFinset_setOf, ne_eq] at h
    rw [← h, ← Finset.sum_subset
      (Finset.subset_univ (Finset.filter (fun (j : Fin n) ↦ ¬(f.toFun j) x = 0) Finset.univ))
      (by intro j hju hj
          simp only [Finset.mem_filter, Finset.mem_univ, true_and, Decidable.not_not] at hj
          exact hj)]
    /-
      case h.refine_1
      X : Type v
      inst✝² : TopologicalSpace X
      inst✝¹ : T2Space X
      inst✝ : LocallyCompactSpace X
      n : Nat
      t : Set X
      s : Fin n → Set X
      hs : ∀ (i : Fin n), IsOpen (s i)
      htcp : IsCompact t
      hst : HasSubset.Subset t (Set.iUnion fun i => s i)
      f : PartitionOfUnity (Fin n) X t
      hfsub : f.IsSubordinate s
      hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
      x : X
      hx : Membership.mem t x
      h : Eq ((Finset.filter (fun x_1 => Not (Eq ((f.toFun x_1) x) 0)) Finset.univ). …
      ⊢ Eq ((Finset.filter (fun j => Not (Eq ((f.toFun j) x) 0)) Finset.univ).sum fu …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case h.refine_2
    X : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    n : Nat
    t : Set X
    s : Fin n → Set X
    hs : ∀ (i : Fin n), IsOpen (s i)
    htcp : IsCompact t
    hst : HasSubset.Subset t (Set.iUnion fun i => s i)
    f : PartitionOfUnity (Fin n) X t
    hfsub : f.IsSubordinate s
    hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
    ⊢ ∀ (i : Fin n) (x : X), Membership.mem (Set.Icc 0 1) ((f i) x)
  -/
  intro i x
  /-
    case h.refine_2
    X : Type v
    inst✝² : TopologicalSpace X
    inst✝¹ : T2Space X
    inst✝ : LocallyCompactSpace X
    n : Nat
    t : Set X
    s : Fin n → Set X
    hs : ∀ (i : Fin n), IsOpen (s i)
    htcp : IsCompact t
    hst : HasSubset.Subset t (Set.iUnion fun i => s i)
    f : PartitionOfUnity (Fin n) X t
    hfsub : f.IsSubordinate s
    hfcp : ∀ (i : Fin n), HasCompactSupport ⇑(f i)
    i : Fin n
    x : X
    ⊢ Membership.mem (Set.Icc 0 1) ((f i) x)
  -/
  exact ⟨f.nonneg i x, PartitionOfUnity.le_one f i x⟩
  /-
    🎉 no goals
  -/

