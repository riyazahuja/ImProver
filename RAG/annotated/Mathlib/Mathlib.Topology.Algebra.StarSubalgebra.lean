instance [TopologicalSemiring A] (s : StarSubalgebra R A) : TopologicalSemiring s :=
  s.toSubalgebra.topologicalSemiring


/-- The `StarSubalgebra.inclusion` of a star subalgebra is an `Embedding`. -/
lemma isEmbedding_inclusion {S₁ S₂ : StarSubalgebra R A} (h : S₁ ≤ S₂) :
    IsEmbedding (inclusion h) where
  eq_induced := Eq.symm induced_compose
  injective := Subtype.map_injective h Function.injective_id


@[deprecated (since := "2024-10-26")]
alias embedding_inclusion := isEmbedding_inclusion


/-- The `StarSubalgebra.inclusion` of a closed star subalgebra is a `IsClosedEmbedding`. -/
theorem isClosedEmbedding_inclusion {S₁ S₂ : StarSubalgebra R A} (h : S₁ ≤ S₂)
    (hS₁ : IsClosed (S₁ : Set A)) : IsClosedEmbedding (inclusion h) :=
  { IsEmbedding.inclusion h with
    isClosed_range := isClosed_induced_iff.2
      ⟨S₁, hS₁, by
          /-
            R : Type u_1
            A : Type u_2
            inst✝⁶ : CommSemiring R
            inst✝⁵ : StarRing R
            inst✝⁴ : TopologicalSpace A
            inst✝³ : Semiring A
            inst✝² : Algebra R A
            inst✝¹ : StarRing A
            inst✝ : StarModule R A
            S₁ S₂ : StarSubalgebra R A
            h : LE.le S₁ S₂
            hS₁ : IsClosed ↑S₁
            ⊢ Eq (Set.preimage Subtype.val ↑S₁) (Set.range ⇑(StarSubalgebra.inclusion h))
          -/
          convert (Set.range_subtype_map id _).symm
            /-
              case h.e'_2.h.e'_4
              R : Type u_1
              A : Type u_2
              inst✝⁶ : CommSemiring R
              inst✝⁵ : StarRing R
              inst✝⁴ : TopologicalSpace A
              inst✝³ : Semiring A
              inst✝² : Algebra R A
              inst✝¹ : StarRing A
              inst✝ : StarModule R A
              S₁ S₂ : StarSubalgebra R A
              h : LE.le S₁ S₂
              hS₁ : IsClosed ↑S₁
              ⊢ Eq (↑S₁) (Set.image id (setOf fun x => Membership.mem S₁ x))
            -/
          · rw [Set.image_id]; rfl
                               /-
                                 🎉 no goals
                               -/
            /-
              case convert_4
              R : Type u_1
              A : Type u_2
              inst✝⁶ : CommSemiring R
              inst✝⁵ : StarRing R
              inst✝⁴ : TopologicalSpace A
              inst✝³ : Semiring A
              inst✝² : Algebra R A
              inst✝¹ : StarRing A
              inst✝ : StarModule R A
              S₁ S₂ : StarSubalgebra R A
              h : LE.le S₁ S₂
              hS₁ : IsClosed ↑S₁
              ⊢ ∀ (x : A), Membership.mem S₁ x → Membership.mem S₂ (id x)
            -/
          · intro _ h'
            /-
              case convert_4
              R : Type u_1
              A : Type u_2
              inst✝⁶ : CommSemiring R
              inst✝⁵ : StarRing R
              inst✝⁴ : TopologicalSpace A
              inst✝³ : Semiring A
              inst✝² : Algebra R A
              inst✝¹ : StarRing A
              inst✝ : StarModule R A
              S₁ S₂ : StarSubalgebra R A
              h : LE.le S₁ S₂
              hS₁ : IsClosed ↑S₁
              x✝ : A
              h' : Membership.mem S₁ x✝
              ⊢ Membership.mem S₂ (id x✝)
            -/
            apply h h' ⟩ }
            /-
              🎉 no goals
            -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_inclusion := isClosedEmbedding_inclusion


/-- The closure of a star subalgebra in a topological star algebra as a star subalgebra. -/
def topologicalClosure (s : StarSubalgebra R A) : StarSubalgebra R A :=
  {
    s.toSubalgebra.topologicalClosure with
    carrier := closure (s : Set A)
    star_mem' := fun ha =>
      map_mem_closure continuous_star ha fun x => (star_mem : x ∈ s → star x ∈ s) }


theorem topologicalClosure_toSubalgebra_comm (s : StarSubalgebra R A) :
    s.topologicalClosure.toSubalgebra = s.toSubalgebra.topologicalClosure :=
  SetLike.coe_injective rfl


@[simp]
theorem topologicalClosure_coe (s : StarSubalgebra R A) :
    (s.topologicalClosure : Set A) = closure (s : Set A) :=
  rfl


theorem le_topologicalClosure (s : StarSubalgebra R A) : s ≤ s.topologicalClosure :=
  subset_closure


theorem isClosed_topologicalClosure (s : StarSubalgebra R A) :
    IsClosed (s.topologicalClosure : Set A) :=
  isClosed_closure


instance {A : Type*} [UniformSpace A] [CompleteSpace A] [Semiring A] [StarRing A]
    [TopologicalSemiring A] [ContinuousStar A] [Algebra R A] [StarModule R A]
    {S : StarSubalgebra R A} : CompleteSpace S.topologicalClosure :=
  isClosed_closure.completeSpace_coe


theorem topologicalClosure_minimal {s t : StarSubalgebra R A} (h : s ≤ t)
    (ht : IsClosed (t : Set A)) : s.topologicalClosure ≤ t :=
  closure_minimal h ht


theorem topologicalClosure_mono : Monotone (topologicalClosure : _ → StarSubalgebra R A) :=
  fun _ S₂ h =>
  topologicalClosure_minimal (h.trans <| le_topologicalClosure S₂) (isClosed_topologicalClosure S₂)


theorem topologicalClosure_map_le [StarModule R B] [TopologicalSemiring B] [ContinuousStar B]
    (s : StarSubalgebra R A) (φ : A →⋆ₐ[R] B) (hφ : IsClosedMap φ) :
    (map φ s).topologicalClosure ≤ map φ s.topologicalClosure :=
  hφ.closure_image_subset _


theorem map_topologicalClosure_le [StarModule R B] [TopologicalSemiring B] [ContinuousStar B]
    (s : StarSubalgebra R A) (φ : A →⋆ₐ[R] B) (hφ : Continuous φ) :
    map φ s.topologicalClosure ≤ (map φ s).topologicalClosure :=
  image_closure_subset_closure_image hφ


theorem topologicalClosure_map [StarModule R B] [TopologicalSemiring B] [ContinuousStar B]
    (s : StarSubalgebra R A) (φ : A →⋆ₐ[R] B) (hφ : IsClosedEmbedding φ) :
    (map φ s).topologicalClosure = map φ s.topologicalClosure :=
  SetLike.coe_injective <| hφ.closure_image_eq _


theorem _root_.Subalgebra.topologicalClosure_star_comm (s : Subalgebra R A) :
    (star s).topologicalClosure = star s.topologicalClosure := by
  suffices ∀ t : Subalgebra R A, (star t).topologicalClosure ≤ star t.topologicalClosure from
    le_antisymm (this s) (by simpa only [star_star] using Subalgebra.star_mono (this (star s)))
  exact fun t => (star t).topologicalClosure_minimal (Subalgebra.star_mono subset_closure)
    (isClosed_closure.preimage continuous_star)


/-- If a star subalgebra of a topological star algebra is commutative, then so is its topological
closure. See note [reducible non-instances]. -/
abbrev commSemiringTopologicalClosure [T2Space A] (s : StarSubalgebra R A)
    (hs : ∀ x y : s, x * y = y * x) : CommSemiring s.topologicalClosure :=
  s.toSubalgebra.commSemiringTopologicalClosure hs


/-- If a star subalgebra of a topological star algebra is commutative, then so is its topological
closure. See note [reducible non-instances]. -/
abbrev commRingTopologicalClosure {R A} [CommRing R] [StarRing R] [TopologicalSpace A] [Ring A]
    [Algebra R A] [StarRing A] [StarModule R A] [TopologicalRing A] [ContinuousStar A] [T2Space A]
    (s : StarSubalgebra R A) (hs : ∀ x y : s, x * y = y * x) : CommRing s.topologicalClosure :=
  s.toSubalgebra.commRingTopologicalClosure hs


/-- Continuous `StarAlgHom`s from the topological closure of a `StarSubalgebra` whose
compositions with the `StarSubalgebra.inclusion` map agree are, in fact, equal. -/
theorem _root_.StarAlgHom.ext_topologicalClosure [T2Space B] {S : StarSubalgebra R A}
    {φ ψ : S.topologicalClosure →⋆ₐ[R] B} (hφ : Continuous φ) (hψ : Continuous ψ)
    (h :
      φ.comp (inclusion (le_topologicalClosure S)) = ψ.comp (inclusion (le_topologicalClosure S))) :
    φ = ψ := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹³ : CommSemiring R
    inst✝¹² : StarRing R
    inst✝¹¹ : TopologicalSpace A
    inst✝¹⁰ : Semiring A
    inst✝⁹ : Algebra R A
    inst✝⁸ : StarRing A
    inst✝⁷ : StarModule R A
    inst✝⁶ : TopologicalSemiring A
    inst✝⁵ : ContinuousStar A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : Semiring B
    inst✝² : Algebra R B
    inst✝¹ : StarRing B
    inst✝ : T2Space B
    S : StarSubalgebra R A
    φ ψ : StarAlgHom R (Subtype fun x => Membership.mem S.topologicalClosure x) B
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ.comp (StarSubalgebra.inclusion ⋯)) (ψ.comp (StarSubalgebra.inclusion …
    ⊢ Eq φ ψ
  -/
  rw [DFunLike.ext'_iff]
  have : Dense (Set.range <| inclusion (le_topologicalClosure S)) := by
    refine IsInducing.subtypeVal.dense_iff.2 fun x => ?_
    convert show ↑x ∈ closure (S : Set A) from x.prop
    rw [← Set.range_comp]
    exact
      Set.ext fun y =>
        ⟨by
          rintro ⟨y, rfl⟩
          exact y.prop, fun hy => ⟨⟨y, hy⟩, rfl⟩⟩
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹³ : CommSemiring R
    inst✝¹² : StarRing R
    inst✝¹¹ : TopologicalSpace A
    inst✝¹⁰ : Semiring A
    inst✝⁹ : Algebra R A
    inst✝⁸ : StarRing A
    inst✝⁷ : StarModule R A
    inst✝⁶ : TopologicalSemiring A
    inst✝⁵ : ContinuousStar A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : Semiring B
    inst✝² : Algebra R B
    inst✝¹ : StarRing B
    inst✝ : T2Space B
    S : StarSubalgebra R A
    φ ψ : StarAlgHom R (Subtype fun x => Membership.mem S.topologicalClosure x) B
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ.comp (StarSubalgebra.inclusion ⋯)) (ψ.comp (StarSubalgebra.inclusion …
    this : Dense (Set.range ⇑(StarSubalgebra.inclusion ⋯))
    ⊢ Eq ⇑φ ⇑ψ
  -/
  refine Continuous.ext_on this hφ hψ ?_
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹³ : CommSemiring R
    inst✝¹² : StarRing R
    inst✝¹¹ : TopologicalSpace A
    inst✝¹⁰ : Semiring A
    inst✝⁹ : Algebra R A
    inst✝⁸ : StarRing A
    inst✝⁷ : StarModule R A
    inst✝⁶ : TopologicalSemiring A
    inst✝⁵ : ContinuousStar A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : Semiring B
    inst✝² : Algebra R B
    inst✝¹ : StarRing B
    inst✝ : T2Space B
    S : StarSubalgebra R A
    φ ψ : StarAlgHom R (Subtype fun x => Membership.mem S.topologicalClosure x) B
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ.comp (StarSubalgebra.inclusion ⋯)) (ψ.comp (StarSubalgebra.inclusion …
    this : Dense (Set.range ⇑(StarSubalgebra.inclusion ⋯))
    ⊢ Set.EqOn (⇑φ) (⇑ψ) (Set.range ⇑(StarSubalgebra.inclusion ⋯))
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹³ : CommSemiring R
    inst✝¹² : StarRing R
    inst✝¹¹ : TopologicalSpace A
    inst✝¹⁰ : Semiring A
    inst✝⁹ : Algebra R A
    inst✝⁸ : StarRing A
    inst✝⁷ : StarModule R A
    inst✝⁶ : TopologicalSemiring A
    inst✝⁵ : ContinuousStar A
    inst✝⁴ : TopologicalSpace B
    inst✝³ : Semiring B
    inst✝² : Algebra R B
    inst✝¹ : StarRing B
    inst✝ : T2Space B
    S : StarSubalgebra R A
    φ ψ : StarAlgHom R (Subtype fun x => Membership.mem S.topologicalClosure x) B
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ.comp (StarSubalgebra.inclusion ⋯)) (ψ.comp (StarSubalgebra.inclusion …
    this : Dense (Set.range ⇑(StarSubalgebra.inclusion ⋯))
    x : Subtype fun x => Membership.mem S x
    ⊢ Eq (φ ((StarSubalgebra.inclusion ⋯) x)) (ψ ((StarSubalgebra.inclusion ⋯) x))
  -/
  simpa only using DFunLike.congr_fun h x
  /-
    🎉 no goals
  -/


theorem _root_.StarAlgHomClass.ext_topologicalClosure [T2Space B] {F : Type*}
    {S : StarSubalgebra R A} [FunLike F S.topologicalClosure B]
    [AlgHomClass F R S.topologicalClosure B] [StarHomClass F S.topologicalClosure B] {φ ψ : F}
    (hφ : Continuous φ) (hψ : Continuous ψ) (h : ∀ x : S,
        φ (inclusion (le_topologicalClosure S) x) = ψ ((inclusion (le_topologicalClosure S)) x)) :
    φ = ψ := by
  -- Porting note: an intervening coercion seems to have appeared since ML3
  have : (φ : S.topologicalClosure →⋆ₐ[R] B) = (ψ : S.topologicalClosure →⋆ₐ[R] B) := by
    refine StarAlgHom.ext_topologicalClosure (R := R) (A := A) (B := B) hφ hψ (StarAlgHom.ext ?_)
    simpa only [StarAlgHom.coe_comp, StarAlgHom.coe_coe] using h
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : StarRing R
    inst✝¹⁴ : TopologicalSpace A
    inst✝¹³ : Semiring A
    inst✝¹² : Algebra R A
    inst✝¹¹ : StarRing A
    inst✝¹⁰ : StarModule R A
    inst✝⁹ : TopologicalSemiring A
    inst✝⁸ : ContinuousStar A
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : Semiring B
    inst✝⁵ : Algebra R B
    inst✝⁴ : StarRing B
    inst✝³ : T2Space B
    F : Type u_4
    S : StarSubalgebra R A
    inst✝² : FunLike F (Subtype fun x => Membership.mem S.topologicalClosure x) B
    inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem S.topologicalClosure …
    inst✝ : StarHomClass F (Subtype fun x => Membership.mem S.topologicalClosure x …
    φ ψ : F
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : ∀ (x : Subtype fun x => Membership.mem S x), Eq (φ ((StarSubalgebra.inclus …
    this : Eq ↑φ ↑ψ
    ⊢ Eq φ ψ
  -/
  rw [DFunLike.ext'_iff, ← StarAlgHom.coe_coe]
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : StarRing R
    inst✝¹⁴ : TopologicalSpace A
    inst✝¹³ : Semiring A
    inst✝¹² : Algebra R A
    inst✝¹¹ : StarRing A
    inst✝¹⁰ : StarModule R A
    inst✝⁹ : TopologicalSemiring A
    inst✝⁸ : ContinuousStar A
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : Semiring B
    inst✝⁵ : Algebra R B
    inst✝⁴ : StarRing B
    inst✝³ : T2Space B
    F : Type u_4
    S : StarSubalgebra R A
    inst✝² : FunLike F (Subtype fun x => Membership.mem S.topologicalClosure x) B
    inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem S.topologicalClosure …
    inst✝ : StarHomClass F (Subtype fun x => Membership.mem S.topologicalClosure x …
    φ ψ : F
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : ∀ (x : Subtype fun x => Membership.mem S x), Eq (φ ((StarSubalgebra.inclus …
    this : Eq ↑φ ↑ψ
    ⊢ Eq ⇑↑φ ⇑ψ
  -/
  apply congrArg _ this
  /-
    🎉 no goals
  -/


/-- The topological closure of the star subalgebra generated by a single element. -/
def elemental (x : A) : StarSubalgebra R A :=
  (adjoin R ({x} : Set A)).topologicalClosure


@[deprecated (since := "2024-11-05")] alias _root_.elementalStarAlgebra := elemental


@[aesop safe apply (rule_sets := [SetLike])]
theorem self_mem (x : A) : x ∈ elemental R x :=
  le_topologicalClosure _ (self_mem_adjoin_singleton R x)


@[deprecated (since := "2024-11-05")] alias _root_.elementalStarAlgebra.self_mem := self_mem


theorem star_self_mem (x : A) : star x ∈ elemental R x :=
  star_mem <| self_mem R x


@[deprecated (since := "2024-11-05")]
alias _root_.elementalStarAlgebra.star_self_mem := star_self_mem


/-- The `elemental` star subalgebra generated by a normal element is commutative. -/
instance [T2Space A] {x : A} [IsStarNormal x] : CommSemiring (elemental R x) :=
  StarSubalgebra.commSemiringTopologicalClosure _ mul_comm


/-- The `elemental` generated by a normal element is commutative. -/
instance {R A} [CommRing R] [StarRing R] [TopologicalSpace A] [Ring A] [Algebra R A] [StarRing A]
    [StarModule R A] [TopologicalRing A] [ContinuousStar A] [T2Space A] {x : A} [IsStarNormal x] :
    CommRing (elemental R x) :=
  StarSubalgebra.commRingTopologicalClosure _ mul_comm


theorem isClosed (x : A) : IsClosed (elemental R x : Set A) :=
  isClosed_closure


@[deprecated (since := "2024-11-05")] alias _root_.elementalStarAlgebra.isClosed := isClosed


instance {A : Type*} [UniformSpace A] [CompleteSpace A] [Semiring A] [StarRing A]
    [TopologicalSemiring A] [ContinuousStar A] [Algebra R A] [StarModule R A] (x : A) :
    CompleteSpace (elemental R x) :=
  isClosed_closure.completeSpace_coe


variable {R} in
theorem le_of_mem {S : StarSubalgebra R A} (hS : IsClosed (S : Set A)) {x : A}
    (hx : x ∈ S) : elemental R x ≤ S :=
  topologicalClosure_minimal (adjoin_le <| Set.singleton_subset_iff.2 hx) hS


variable {R} in
theorem le_iff_mem {x : A} {s : StarSubalgebra R A} (hs : IsClosed (s : Set A)) :
    elemental R x ≤ s ↔ x ∈ s :=
  ⟨fun h ↦ h (self_mem R x), fun h ↦ le_of_mem hs h⟩


@[deprecated (since := "2024-11-05")]
alias _root_.elementalStarAlgebra.le_of_isClosed_of_mem := le_of_mem


/-- The coercion from an elemental algebra to the full algebra as a `IsClosedEmbedding`. -/
theorem isClosedEmbedding_coe (x : A) : IsClosedEmbedding ((↑) : elemental R x → A) where
  eq_induced := rfl
  injective := Subtype.coe_injective
                       /-
                         R : Type u_1
                         A : Type u_2
                         inst✝⁸ : CommSemiring R
                         inst✝⁷ : StarRing R
                         inst✝⁶ : TopologicalSpace A
                         inst✝⁵ : Semiring A
                         inst✝⁴ : StarRing A
                         inst✝³ : TopologicalSemiring A
                         inst✝² : ContinuousStar A
                         inst✝¹ : Algebra R A
                         inst✝ : StarModule R A
                         x : A
                         ⊢ IsClosed (Set.range Subtype.val)
                       -/
  isClosed_range := by simpa using isClosed R x
                       /-
                         🎉 no goals
                       -/


@[deprecated (since := "2024-11-05")]
alias _root_.elementalStarAlgebra.isClosedEmbedding_coe := isClosedEmbedding_coe

@[deprecated (since := "2024-10-20")]
alias _root_.elementalStarAlgebra.closedEmbedding_coe := isClosedEmbedding_coe


@[elab_as_elim]
theorem induction_on {x y : A}
    (hy : y ∈ elemental R x) {P : (u : A) → u ∈ elemental R x → Prop}
    (self : P x (self_mem R x)) (star_self : P (star x) (star_self_mem R x))
    (algebraMap : ∀ r, P (algebraMap R A r) (_root_.algebraMap_mem _ r))
    (add : ∀ u hu v hv, P u hu → P v hv → P (u + v) (add_mem hu hv))
    (mul : ∀ u hu v hv, P u hu → P v hv → P (u * v) (mul_mem hu hv))
    (closure : ∀ s : Set A, (hs : s ⊆ elemental R x) → (∀ u, (hu : u ∈ s) →
      P u (hs hu)) → ∀ v, (hv : v ∈ closure s) → P v (closure_minimal hs (isClosed R x) hv)) :
    P y hy := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Semiring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSemiring A
    inst✝² : ContinuousStar A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    x y : A
    hy : Membership.mem (StarAlgebra.elemental R x) y
    P : (u : A) → Membership.mem (StarAlgebra.elemental R x) u → Prop
    self : P x ⋯
    star_self : P (Star.star x) ⋯
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R A) r) ⋯
    add : ∀ (u : A) (hu : Membership.mem (StarAlgebra.elemental R x) u) (v : A) (h …
    mul : ∀ (u : A) (hu : Membership.mem (StarAlgebra.elemental R x) u) (v : A) (h …
    closure : ∀ (s : Set A) (hs : HasSubset.Subset s ↑(StarAlgebra.elemental R x)) …
    ⊢ P y hy
  -/
  apply closure (adjoin R {x} : Set A) subset_closure (fun y hy ↦ ?_) y hy
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁸ : CommSemiring R
    inst✝⁷ : StarRing R
    inst✝⁶ : TopologicalSpace A
    inst✝⁵ : Semiring A
    inst✝⁴ : StarRing A
    inst✝³ : TopologicalSemiring A
    inst✝² : ContinuousStar A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    x y✝ : A
    hy✝ : Membership.mem (StarAlgebra.elemental R x) y✝
    P : (u : A) → Membership.mem (StarAlgebra.elemental R x) u → Prop
    self : P x ⋯
    star_self : P (Star.star x) ⋯
    algebraMap : ∀ (r : R), P ((_root_.algebraMap R A) r) ⋯
    add : ∀ (u : A) (hu : Membership.mem (StarAlgebra.elemental R x) u) (v : A) (h …
    mul : ∀ (u : A) (hu : Membership.mem (StarAlgebra.elemental R x) u) (v : A) (h …
    closure : ∀ (s : Set A) (hs : HasSubset.Subset s ↑(StarAlgebra.elemental R x)) …
    y : A
    hy : Membership.mem (↑(StarAlgebra.adjoin R (Singleton.singleton x))) y
    ⊢ P y ⋯
  -/
  rw [SetLike.mem_coe, ← mem_toSubalgebra, adjoin_toSubalgebra] at hy
  induction hy using Algebra.adjoin_induction with
  | mem u hu =>
    obtain ((rfl : u = x) | (hu : star u = x)) := by simpa using hu
    · exact self
    · simp_rw [← hu, star_star] at star_self
      exact star_self
  | algebraMap r => exact algebraMap r
  | add u v hu_mem hv_mem hu hv =>
    exact add u (subset_closure hu_mem) v (subset_closure hv_mem) (hu hu_mem) (hv hv_mem)
  | mul u v hu_mem hv_mem hu hv =>
    exact mul u (subset_closure hu_mem) v (subset_closure hv_mem) (hu hu_mem) (hv hv_mem)


@[deprecated (since := "2024-11-05")]
alias _root_.elementalStarAlgebra.induction_on := induction_on


theorem starAlgHomClass_ext [T2Space B] {F : Type*} {a : A}
    [FunLike F (elemental R a) B] [AlgHomClass F R _ B] [StarHomClass F _ B]
    {φ ψ : F} (hφ : Continuous φ)
    (hψ : Continuous ψ) (h : φ ⟨a, self_mem R a⟩ = ψ ⟨a, self_mem R a⟩) : φ = ψ := by
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : StarRing R
    inst✝¹⁴ : TopologicalSpace A
    inst✝¹³ : Semiring A
    inst✝¹² : StarRing A
    inst✝¹¹ : TopologicalSemiring A
    inst✝¹⁰ : ContinuousStar A
    inst✝⁹ : Algebra R A
    inst✝⁸ : StarModule R A
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : Semiring B
    inst✝⁵ : StarRing B
    inst✝⁴ : Algebra R B
    inst✝³ : T2Space B
    F : Type u_4
    a : A
    inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.elemental R a …
    inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.element …
    inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.elemental …
    φ ψ : F
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ ⟨a, ⋯⟩) (ψ ⟨a, ⋯⟩)
    ⊢ Eq φ ψ
  -/
  refine StarAlgHomClass.ext_topologicalClosure hφ hψ fun x => ?_
  /-
    R : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝¹⁶ : CommSemiring R
    inst✝¹⁵ : StarRing R
    inst✝¹⁴ : TopologicalSpace A
    inst✝¹³ : Semiring A
    inst✝¹² : StarRing A
    inst✝¹¹ : TopologicalSemiring A
    inst✝¹⁰ : ContinuousStar A
    inst✝⁹ : Algebra R A
    inst✝⁸ : StarModule R A
    inst✝⁷ : TopologicalSpace B
    inst✝⁶ : Semiring B
    inst✝⁵ : StarRing B
    inst✝⁴ : Algebra R B
    inst✝³ : T2Space B
    F : Type u_4
    a : A
    inst✝² : FunLike F (Subtype fun x => Membership.mem (StarAlgebra.elemental R a …
    inst✝¹ : AlgHomClass F R (Subtype fun x => Membership.mem (StarAlgebra.element …
    inst✝ : StarHomClass F (Subtype fun x => Membership.mem (StarAlgebra.elemental …
    φ ψ : F
    hφ : Continuous ⇑φ
    hψ : Continuous ⇑ψ
    h : Eq (φ ⟨a, ⋯⟩) (ψ ⟨a, ⋯⟩)
    x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R (Singleton.singleton …
    ⊢ Eq (φ ((StarSubalgebra.inclusion ⋯) x)) (ψ ((StarSubalgebra.inclusion ⋯) x))
  -/
  refine adjoin_induction_subtype x ?_ ?_ ?_ ?_ ?_
  exacts [fun y hy => by simpa only [Set.mem_singleton_iff.mp hy] using h, fun r => by
    simp only [AlgHomClass.commutes], fun x y hx hy => by simp only [map_add, hx, hy],
    fun x y hx hy => by simp only [map_mul, hx, hy], fun x hx => by simp only [map_star, hx]]


@[deprecated (since := "2024-11-05")]
alias _root_.elementalStarAlgebra.starAlgHomClass_ext := starAlgHomClass_ext


