/-- The `J`-closure of a sieve is the collection of arrows which it covers. -/
@[simps]
def close {X : C} (S : Sieve X) : Sieve X where
  arrows _ f := J₁.Covers S f
  downward_closed hS := J₁.arrow_stable _ _ hS


/-- Any sieve is smaller than its closure. -/
theorem le_close {X : C} (S : Sieve X) : S ≤ J₁.close S :=
  fun _ _ hg => J₁.covering_of_eq_top (S.pullback_eq_top_of_mem hg)


/-- A sieve is closed for the Grothendieck topology if it contains every arrow it covers.
In the case of the usual topology on a topological space, this means that the open cover contains
every open set which it covers.

Note this has no relation to a closed subset of a topological space.
-/
def IsClosed {X : C} (S : Sieve X) : Prop :=
  ∀ ⦃Y : C⦄ (f : Y ⟶ X), J₁.Covers S f → S f


/-- If `S` is `J₁`-closed, then `S` covers exactly the arrows it contains. -/
theorem covers_iff_mem_of_isClosed {X : C} {S : Sieve X} (h : J₁.IsClosed S) {Y : C} (f : Y ⟶ X) :
    J₁.Covers S f ↔ S f :=
  ⟨h _, J₁.arrow_max _ _⟩


/-- Being `J`-closed is stable under pullback. -/
theorem isClosed_pullback {X Y : C} (f : Y ⟶ X) (S : Sieve X) :
    J₁.IsClosed S → J₁.IsClosed (S.pullback f) :=
                                  /-
                                    C : Type u
                                    inst✝ : CategoryTheory.Category.{v, u} C
                                    J₁ : CategoryTheory.GrothendieckTopology C
                                    X Y : C
                                    f : Quiver.Hom Y X
                                    S : CategoryTheory.Sieve X
                                    hS : J₁.IsClosed S
                                    Z : C
                                    g : Quiver.Hom Z Y
                                    hg : J₁.Covers (CategoryTheory.Sieve.pullback f S) g
                                    ⊢ J₁.Covers S (CategoryTheory.CategoryStruct.comp g f)
                                  -/
  fun hS Z g hg => hS (g ≫ f) (by rwa [J₁.covers_iff, Sieve.pullback_comp])
                                  /-
                                    🎉 no goals
                                  -/


/-- The closure of a sieve `S` is the largest closed sieve which contains `S` (justifying the name
"closure").
-/
theorem le_close_of_isClosed {X : C} {S T : Sieve X} (h : S ≤ T) (hT : J₁.IsClosed T) :
    J₁.close S ≤ T :=
  fun _ f hf => hT _ (J₁.superset_covering (Sieve.pullback_monotone f h) hf)


/-- The closure of a sieve is closed. -/
theorem close_isClosed {X : C} (S : Sieve X) : J₁.IsClosed (J₁.close S) :=
  fun _ g hg => J₁.arrow_trans g _ S hg fun _ hS => hS


/-- A Grothendieck topology induces a natural family of closure operators on sieves. -/
@[simps! isClosed]
def closureOperator (X : C) : ClosureOperator (Sieve X) :=
  .ofPred J₁.close J₁.IsClosed J₁.le_close J₁.close_isClosed fun _ _ ↦ J₁.le_close_of_isClosed


/-- The sieve `S` is closed iff its closure is equal to itself. -/
theorem isClosed_iff_close_eq_self {X : C} (S : Sieve X) : J₁.IsClosed S ↔ J₁.close S = S :=
  (J₁.closureOperator _).isClosed_iff


theorem close_eq_self_of_isClosed {X : C} {S : Sieve X} (hS : J₁.IsClosed S) : J₁.close S = S :=
  (J₁.isClosed_iff_close_eq_self S).1 hS


/-- Closing under `J` is stable under pullback. -/
theorem pullback_close {X Y : C} (f : Y ⟶ X) (S : Sieve X) :
    J₁.close (S.pullback f) = (J₁.close S).pullback f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    X Y : C
    f : Quiver.Hom Y X
    S : CategoryTheory.Sieve X
    ⊢ Eq (J₁.close (CategoryTheory.Sieve.pullback f S)) (CategoryTheory.Sieve.pull …
  -/
  apply le_antisymm
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Sieve X
      ⊢ LE.le (J₁.close (CategoryTheory.Sieve.pullback f S)) (CategoryTheory.Sieve.p …
    -/
  · refine J₁.le_close_of_isClosed (Sieve.pullback_monotone _ (J₁.le_close S)) ?_
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Sieve X
      ⊢ J₁.IsClosed (CategoryTheory.Sieve.pullback f (J₁.close S))
    -/
    apply J₁.isClosed_pullback _ _ (J₁.close_isClosed _)
    /-
      🎉 no goals
    -/
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Sieve X
      ⊢ LE.le (CategoryTheory.Sieve.pullback f (J₁.close S)) (J₁.close (CategoryTheo …
    -/
  · intro Z g hg
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Sieve X
      Z : C
      g : Quiver.Hom Z Y
      hg : (CategoryTheory.Sieve.pullback f (J₁.close S)).arrows g
      ⊢ (J₁.close (CategoryTheory.Sieve.pullback f S)).arrows g
    -/
    change _ ∈ J₁ _
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Sieve X
      Z : C
      g : Quiver.Hom Z Y
      hg : (CategoryTheory.Sieve.pullback f (J₁.close S)).arrows g
      ⊢ Membership.mem (J₁ Z) (CategoryTheory.Sieve.pullback g (CategoryTheory.Sieve …
    -/
    rw [← Sieve.pullback_comp]
    /-
      case a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X Y : C
      f : Quiver.Hom Y X
      S : CategoryTheory.Sieve X
      Z : C
      g : Quiver.Hom Z Y
      hg : (CategoryTheory.Sieve.pullback f (J₁.close S)).arrows g
      ⊢ Membership.mem (J₁ Z) (CategoryTheory.Sieve.pullback (CategoryTheory.Categor …
    -/
    apply hg
    /-
      🎉 no goals
    -/


@[mono]
theorem monotone_close {X : C} : Monotone (J₁.close : Sieve X → Sieve X) :=
  (J₁.closureOperator _).monotone


@[simp]
theorem close_close {X : C} (S : Sieve X) : J₁.close (J₁.close S) = J₁.close S :=
  (J₁.closureOperator _).idempotent _


/--
The sieve `S` is in the topology iff its closure is the maximal sieve. This shows that the closure
operator determines the topology.
-/
theorem close_eq_top_iff_mem {X : C} (S : Sieve X) : J₁.close S = ⊤ ↔ S ∈ J₁ X := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Eq (J₁.close S) Top.top) (Membership.mem (J₁ X) S)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ Eq (J₁.close S) Top.top → Membership.mem (J₁ X) S
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      h : Eq (J₁.close S) Top.top
      ⊢ Membership.mem (J₁ X) S
    -/
    apply J₁.transitive (J₁.top_mem X)
    /-
      case mp.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      h : Eq (J₁.close S) Top.top
      ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, Top.top.arrows f → Membership.mem (J₁ Y) (Ca …
    -/
    intro Y f hf
    /-
      case mp.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      h : Eq (J₁.close S) Top.top
      Y : C
      f : Quiver.Hom Y X
      hf : Top.top.arrows f
      ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f S)
    -/
    change J₁.close S f
    /-
      case mp.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      h : Eq (J₁.close S) Top.top
      Y : C
      f : Quiver.Hom Y X
      hf : Top.top.arrows f
      ⊢ (J₁.close S).arrows f
    -/
    rwa [h]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      ⊢ Membership.mem (J₁ X) S → Eq (J₁.close S) Top.top
    -/
  · intro hS
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      ⊢ Eq (J₁.close S) Top.top
    -/
    rw [eq_top_iff]
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      ⊢ LE.le Top.top (J₁.close S)
    -/
    intro Y f _
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      Y : C
      f : Quiver.Hom Y X
      a✝ : Top.top.arrows f
      ⊢ (J₁.close S).arrows f
    -/
    apply J₁.pullback_stable _ hS
    /-
      🎉 no goals
    -/


/--
The presheaf sending each object to the set of `J`-closed sieves on it. This presheaf is a `J`-sheaf
(and will turn out to be a subobject classifier for the category of `J`-sheaves).
-/
@[simps]
def Functor.closedSieves : Cᵒᵖ ⥤ Type max v u where
  obj X := { S : Sieve X.unop // J₁.IsClosed S }
  map f S := ⟨S.1.pullback f.unop, J₁.isClosed_pullback f.unop _ S.2⟩


/-- The presheaf of `J`-closed sieves is a `J`-sheaf.
The proof of this is adapted from [MM92], Chapter III, Section 7, Lemma 1.
-/
theorem classifier_isSheaf : Presieve.IsSheaf J₁ (Functor.closedSieves J₁) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    ⊢ CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₁)
  -/
  intro X S hS
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    ⊢ CategoryTheory.Presieve.IsSheafFor (CategoryTheory.Functor.closedSieves J₁)  …
  -/
  rw [← Presieve.isSeparatedFor_and_exists_isAmalgamation_iff_isSheafFor]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    ⊢ And (CategoryTheory.Presieve.IsSeparatedFor (CategoryTheory.Functor.closedSi …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      ⊢ CategoryTheory.Presieve.IsSeparatedFor (CategoryTheory.Functor.closedSieves  …
    -/
  · rintro x ⟨M, hM⟩ ⟨N, hN⟩ hM₂ hN₂
    /-
      case refine_1.mk.mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      M : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hM : J₁.IsClosed M
      N : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hN : J₁.IsClosed N
      hM₂ : x.IsAmalgamation ⟨M, hM⟩
      hN₂ : x.IsAmalgamation ⟨N, hN⟩
      ⊢ Eq ⟨M, hM⟩ ⟨N, hN⟩
    -/
    simp only [Functor.closedSieves_obj]
    /-
      case refine_1.mk.mk
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      M : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hM : J₁.IsClosed M
      N : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hN : J₁.IsClosed N
      hM₂ : x.IsAmalgamation ⟨M, hM⟩
      hN₂ : x.IsAmalgamation ⟨N, hN⟩
      ⊢ Eq ⟨M, hM⟩ ⟨N, hN⟩
    -/
    ext Y f
    /-
      case refine_1.mk.mk.a.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      M : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hM : J₁.IsClosed M
      N : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hN : J₁.IsClosed N
      hM₂ : x.IsAmalgamation ⟨M, hM⟩
      hN₂ : x.IsAmalgamation ⟨N, hN⟩
      Y : C
      f : Quiver.Hom Y X
      ⊢ Iff ((↑⟨M, hM⟩).arrows f) ((↑⟨N, hN⟩).arrows f)
    -/
    dsimp only [Subtype.coe_mk]
    /-
      case refine_1.mk.mk.a.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      M : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hM : J₁.IsClosed M
      N : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hN : J₁.IsClosed N
      hM₂ : x.IsAmalgamation ⟨M, hM⟩
      hN₂ : x.IsAmalgamation ⟨N, hN⟩
      Y : C
      f : Quiver.Hom Y X
      ⊢ Iff (M.arrows f) (N.arrows f)
    -/
    rw [← J₁.covers_iff_mem_of_isClosed hM, ← J₁.covers_iff_mem_of_isClosed hN]
    have q : ∀ ⦃Z : C⦄ (g : Z ⟶ X) (_ : S g), M.pullback g = N.pullback g :=
      fun Z g hg => congr_arg Subtype.val ((hM₂ g hg).trans (hN₂ g hg).symm)
    have MSNS : M ⊓ S = N ⊓ S := by
      ext Z g
      rw [Sieve.inter_apply, Sieve.inter_apply]
      simp only [and_comm]
      apply and_congr_right
      intro hg
      rw [Sieve.pullback_eq_top_iff_mem, Sieve.pullback_eq_top_iff_mem, q g hg]
    /-
      case refine_1.mk.mk.a.h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      M : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hM : J₁.IsClosed M
      N : CategoryTheory.Sieve (Opposite.unop { unop := X })
      hN : J₁.IsClosed N
      hM₂ : x.IsAmalgamation ⟨M, hM⟩
      hN₂ : x.IsAmalgamation ⟨N, hN⟩
      Y : C
      f : Quiver.Hom Y X
      q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
      MSNS : Eq (Min.min M S) (Min.min N S)
      ⊢ Iff (J₁.Covers M f) (J₁.Covers N f)
    -/
    constructor
      /-
        case refine_1.mk.mk.a.h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        ⊢ J₁.Covers M f → J₁.Covers N f
      -/
    · intro hf
      /-
        case refine_1.mk.mk.a.h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers M f
        ⊢ J₁.Covers N f
      -/
      rw [J₁.covers_iff]
      /-
        case refine_1.mk.mk.a.h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers M f
        ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f N)
      -/
      apply J₁.superset_covering (Sieve.pullback_monotone f inf_le_left)
      /-
        case refine_1.mk.mk.a.h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers M f
        ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f (Min.min N ?m.12573))
      -/
      rw [← MSNS]
      /-
        case refine_1.mk.mk.a.h.mp
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers M f
        ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f (Min.min M S))
      -/
      apply J₁.arrow_intersect f M S hf (J₁.pullback_stable _ hS)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.mk.mk.a.h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        ⊢ J₁.Covers N f → J₁.Covers M f
      -/
    · intro hf
      /-
        case refine_1.mk.mk.a.h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers N f
        ⊢ J₁.Covers M f
      -/
      rw [J₁.covers_iff]
      /-
        case refine_1.mk.mk.a.h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers N f
        ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f M)
      -/
      apply J₁.superset_covering (Sieve.pullback_monotone f inf_le_left)
      /-
        case refine_1.mk.mk.a.h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers N f
        ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f (Min.min M ?m.12958))
      -/
      rw [MSNS]
      /-
        case refine_1.mk.mk.a.h.mpr
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ : CategoryTheory.GrothendieckTopology C
        X : C
        S : CategoryTheory.Sieve X
        hS : Membership.mem (J₁ X) S
        x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
        M : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hM : J₁.IsClosed M
        N : CategoryTheory.Sieve (Opposite.unop { unop := X })
        hN : J₁.IsClosed N
        hM₂ : x.IsAmalgamation ⟨M, hM⟩
        hN₂ : x.IsAmalgamation ⟨N, hN⟩
        Y : C
        f : Quiver.Hom Y X
        q : ∀ ⦃Z : C⦄ (g : Quiver.Hom Z X), S.arrows g → Eq (CategoryTheory.Sieve.pull …
        MSNS : Eq (Min.min M S) (Min.min N S)
        hf : J₁.Covers N f
        ⊢ Membership.mem (J₁ Y) (CategoryTheory.Sieve.pullback f (Min.min N S))
      -/
      apply J₁.arrow_intersect f N S hf (J₁.pullback_stable _ hS)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      ⊢ ∀ (x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.clos …
    -/
  · intro x hx
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.Compatible
      ⊢ Exists fun t => x.IsAmalgamation t
    -/
    rw [Presieve.compatible_iff_sieveCompatible] at hx
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      ⊢ Exists fun t => x.IsAmalgamation t
    -/
    let M := Sieve.bind S fun Y f hf => (x f hf).1
    have : ∀ ⦃Y⦄ (f : Y ⟶ X) (hf : S f), M.pullback f = (x f hf).1 := by
      intro Y f hf
      apply le_antisymm
      · rintro Z u ⟨W, g, f', hf', hg : (x f' hf').1 _, c⟩
        rw [Sieve.pullback_eq_top_iff_mem,
          ← show (x (u ≫ f) _).1 = (x f hf).1.pullback u from congr_arg Subtype.val (hx f u hf)]
        conv_lhs => congr; congr; rw [← c] -- Porting note: Originally `simp_rw [← c]`
        rw [show (x (g ≫ f') _).1 = _ from congr_arg Subtype.val (hx f' g hf')]
        apply Sieve.pullback_eq_top_of_mem _ hg
      · apply Sieve.le_pullback_bind S fun Y f hf => (x f hf).1
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      ⊢ Exists fun t => x.IsAmalgamation t
    -/
    refine ⟨⟨_, J₁.close_isClosed M⟩, ?_⟩
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      ⊢ x.IsAmalgamation ⟨J₁.close M, ⋯⟩
    -/
    intro Y f hf
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.Functor.closedSieves J₁).map f.op ⟨J₁.close M, ⋯⟩) (x f  …
    -/
    simp only [Functor.closedSieves_obj]
    /-
      case refine_2
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq ((CategoryTheory.Functor.closedSieves J₁).map f.op ⟨J₁.close M, ⋯⟩) (x f  …
    -/
    ext1
    /-
      case refine_2.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq ↑((CategoryTheory.Functor.closedSieves J₁).map f.op ⟨J₁.close M, ⋯⟩) ↑(x  …
    -/
    dsimp
    /-
      case refine_2.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (CategoryTheory.Sieve.pullback f (J₁.close M)) ↑(x f hf)
    -/
    rw [← J₁.pullback_close, this _ hf]
    /-
      case refine_2.a
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem (J₁ X) S
      x : CategoryTheory.Presieve.FamilyOfElements (CategoryTheory.Functor.closedSie …
      hx : x.SieveCompatible
      M : CategoryTheory.Sieve X := CategoryTheory.Sieve.bind S.arrows fun Y f hf => …
      this : ∀ ⦃Y : C⦄ (f : Quiver.Hom Y X) (hf : S.arrows f), Eq (CategoryTheory.Si …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq (J₁.close ↑(x f hf)) ↑(x f hf)
    -/
    apply le_antisymm (J₁.le_close_of_isClosed le_rfl (x f hf).2) (J₁.le_close _)
    /-
      🎉 no goals
    -/


/-- If presheaf of `J₁`-closed sieves is a `J₂`-sheaf then `J₁ ≤ J₂`. Note the converse is true by
`classifier_isSheaf` and `isSheaf_of_le`.
-/
theorem le_topology_of_closedSieves_isSheaf {J₁ J₂ : GrothendieckTopology C}
    (h : Presieve.IsSheaf J₁ (Functor.closedSieves J₂)) : J₁ ≤ J₂ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    ⊢ LE.le J₁ J₂
  -/
  intro X S hS
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    ⊢ Membership.mem (J₂ X) S
  -/
  rw [← J₂.close_eq_top_iff_mem]
  have : J₂.IsClosed (⊤ : Sieve X) := by
    intro Y f _
    trivial
  suffices (⟨J₂.close S, J₂.close_isClosed S⟩ : Subtype _) = ⟨⊤, this⟩ by
    rw [Subtype.ext_iff] at this
    exact this
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    this : J₂.IsClosed Top.top
    ⊢ Eq ⟨J₂.close S, ⋯⟩ ⟨Top.top, this⟩
  -/
  apply (h S hS).isSeparatedFor.ext
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    this : J₂.IsClosed Top.top
    ⊢ ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Eq ((CategoryTheory.Functor.clo …
  -/
  intro Y f hf
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    this : J₂.IsClosed Top.top
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Eq ((CategoryTheory.Functor.closedSieves J₂).map f.op ⟨J₂.close S, ⋯⟩) ((Cat …
  -/
  simp only [Functor.closedSieves_obj]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    this : J₂.IsClosed Top.top
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Eq ((CategoryTheory.Functor.closedSieves J₂).map f.op ⟨J₂.close S, ⋯⟩) ((Cat …
  -/
  ext1
  /-
    case a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    this : J₂.IsClosed Top.top
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Eq ↑((CategoryTheory.Functor.closedSieves J₂).map f.op ⟨J₂.close S, ⋯⟩) ↑((C …
  -/
  dsimp
  rw [Sieve.pullback_top, ← J₂.pullback_close, S.pullback_eq_top_of_mem hf,
    J₂.close_eq_top_iff_mem]
  /-
    case a
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    h : CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
    X : C
    S : CategoryTheory.Sieve X
    hS : Membership.mem (J₁ X) S
    this : J₂.IsClosed Top.top
    Y : C
    f : Quiver.Hom Y X
    hf : S.arrows f
    ⊢ Membership.mem (J₂ Y) Top.top
  -/
  apply J₂.top_mem
  /-
    🎉 no goals
  -/


/-- If being a sheaf for `J₁` is equivalent to being a sheaf for `J₂`, then `J₁ = J₂`. -/
theorem topology_eq_iff_same_sheaves {J₁ J₂ : GrothendieckTopology C} :
    J₁ = J₂ ↔ ∀ P : Cᵒᵖ ⥤ Type max v u, Presieve.IsSheaf J₁ P ↔ Presieve.IsSheaf J₂ P := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ J₂ : CategoryTheory.GrothendieckTopology C
    ⊢ Iff (Eq J₁ J₂) (∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))) …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      ⊢ Eq J₁ J₂ → ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff …
    -/
  · rintro rfl
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      ⊢ ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (CategoryT …
    -/
    intro P
    /-
      case mp
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ : CategoryTheory.GrothendieckTopology C
      P : CategoryTheory.Functor (Opposite C) (Type (max v u))
      ⊢ Iff (CategoryTheory.Presieve.IsSheaf J₁ P) (CategoryTheory.Presieve.IsSheaf  …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      ⊢ (∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Category …
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
      ⊢ Eq J₁ J₂
    -/
    apply le_antisymm
      /-
        case mpr.a
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ J₂ : CategoryTheory.GrothendieckTopology C
        h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
        ⊢ LE.le J₁ J₂
      -/
    · apply le_topology_of_closedSieves_isSheaf
      /-
        case mpr.a.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ J₂ : CategoryTheory.GrothendieckTopology C
        h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
        ⊢ CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₂)
      -/
      rw [h]
      /-
        case mpr.a.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ J₂ : CategoryTheory.GrothendieckTopology C
        h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
        ⊢ CategoryTheory.Presieve.IsSheaf J₂ (CategoryTheory.Functor.closedSieves J₂)
      -/
      apply classifier_isSheaf
      /-
        🎉 no goals
      -/
      /-
        case mpr.a
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ J₂ : CategoryTheory.GrothendieckTopology C
        h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
        ⊢ LE.le J₂ J₁
      -/
    · apply le_topology_of_closedSieves_isSheaf
      /-
        case mpr.a.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ J₂ : CategoryTheory.GrothendieckTopology C
        h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
        ⊢ CategoryTheory.Presieve.IsSheaf J₂ (CategoryTheory.Functor.closedSieves J₁)
      -/
      rw [← h]
      /-
        case mpr.a.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        J₁ J₂ : CategoryTheory.GrothendieckTopology C
        h : ∀ (P : CategoryTheory.Functor (Opposite C) (Type (max v u))), Iff (Categor …
        ⊢ CategoryTheory.Presieve.IsSheaf J₁ (CategoryTheory.Functor.closedSieves J₁)
      -/
      apply classifier_isSheaf
      /-
        🎉 no goals
      -/


/--
A closure (increasing, inflationary and idempotent) operation on sieves that commutes with pullback
induces a Grothendieck topology.
In fact, such operations are in bijection with Grothendieck topologies.
-/
@[simps]
def topologyOfClosureOperator (c : ∀ X : C, ClosureOperator (Sieve X))
    (hc : ∀ ⦃X Y : C⦄ (f : Y ⟶ X) (S : Sieve X), c _ (S.pullback f) = (c _ S).pullback f) :
    GrothendieckTopology C where
  sieves X := { S | c X S = ⊤ }
  top_mem' X := top_unique ((c X).le_closure _)
  pullback_stable' X Y S f hS := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X Y : C
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hS : Membership.mem ((fun X => setOf fun S => Eq ((c X) S) Top.top) X) S
      ⊢ Membership.mem ((fun X => setOf fun S => Eq ((c X) S) Top.top) Y) (CategoryT …
    -/
    rw [Set.mem_setOf_eq] at hS
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X Y : C
      S : CategoryTheory.Sieve X
      f : Quiver.Hom Y X
      hS : Eq ((c X) S) Top.top
      ⊢ Membership.mem ((fun X => setOf fun S => Eq ((c X) S) Top.top) Y) (CategoryT …
    -/
    rw [Set.mem_setOf_eq, hc, hS, Sieve.pullback_top]
    /-
      🎉 no goals
    -/
  transitive' X S hS R hR := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X : C
      S : CategoryTheory.Sieve X
      hS : Membership.mem ((fun X => setOf fun S => Eq ((c X) S) Top.top) X) S
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X => se …
      ⊢ Membership.mem ((fun X => setOf fun S => Eq ((c X) S) Top.top) X) R
    -/
    rw [Set.mem_setOf_eq] at hS
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X : C
      S : CategoryTheory.Sieve X
      hS : Eq ((c X) S) Top.top
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X => se …
      ⊢ Membership.mem ((fun X => setOf fun S => Eq ((c X) S) Top.top) X) R
    -/
    rw [Set.mem_setOf_eq, ← (c X).idempotent, eq_top_iff, ← hS]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X : C
      S : CategoryTheory.Sieve X
      hS : Eq ((c X) S) Top.top
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X => se …
      ⊢ LE.le ((c X) S) ((c X) ((c X) R))
    -/
    apply (c X).monotone fun Y f hf => _
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X : C
      S : CategoryTheory.Sieve X
      hS : Eq ((c X) S) Top.top
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X => se …
      ⊢ ∀ (Y : C) (f : Quiver.Hom Y X), S.arrows f → ((c X) R).arrows f
    -/
    intros Y f hf
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X : C
      S : CategoryTheory.Sieve X
      hS : Eq ((c X) S) Top.top
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X => se …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ ((c X) R).arrows f
    -/
    rw [Sieve.pullback_eq_top_iff_mem, ← hc]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      J₁ J₂ : CategoryTheory.GrothendieckTopology C
      c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
      hc : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
      X : C
      S : CategoryTheory.Sieve X
      hS : Eq ((c X) S) Top.top
      R : CategoryTheory.Sieve X
      hR : ∀ ⦃Y : C⦄ ⦃f : Quiver.Hom Y X⦄, S.arrows f → Membership.mem ((fun X => se …
      Y : C
      f : Quiver.Hom Y X
      hf : S.arrows f
      ⊢ Eq ((c Y) (CategoryTheory.Sieve.pullback f R)) Top.top
    -/
    apply hR hf
    /-
      🎉 no goals
    -/


/--
The topology given by the closure operator `J.close` on a Grothendieck topology is the same as `J`.
-/
theorem topologyOfClosureOperator_self :
    (topologyOfClosureOperator J₁.closureOperator fun _ _ => J₁.pullback_close) = J₁ := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    ⊢ Eq (CategoryTheory.topologyOfClosureOperator J₁.closureOperator ⋯) J₁
  -/
  ext X S
  /-
    case h.h.h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    J₁ : CategoryTheory.GrothendieckTopology C
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Iff (Membership.mem ((CategoryTheory.topologyOfClosureOperator J₁.closureOpe …
  -/
  apply GrothendieckTopology.close_eq_top_iff_mem
  /-
    🎉 no goals
  -/


theorem topologyOfClosureOperator_close (c : ∀ X : C, ClosureOperator (Sieve X))
    (pb : ∀ ⦃X Y : C⦄ (f : Y ⟶ X) (S : Sieve X), c Y (S.pullback f) = (c X S).pullback f) (X : C)
    (S : Sieve X) : (topologyOfClosureOperator c pb).close S = c X S := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
    pb : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
    X : C
    S : CategoryTheory.Sieve X
    ⊢ Eq ((CategoryTheory.topologyOfClosureOperator c pb).close S) ((c X) S)
  -/
  ext Y f
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
    pb : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    f : Quiver.Hom Y X
    ⊢ Iff (((CategoryTheory.topologyOfClosureOperator c pb).close S).arrows f) ((( …
  -/
  change c _ (Sieve.pullback f S) = ⊤ ↔ c _ S f
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    c : (X : C) → ClosureOperator (CategoryTheory.Sieve X)
    pb : ∀ ⦃X Y : C⦄ (f : Quiver.Hom Y X) (S : CategoryTheory.Sieve X), Eq ((c Y)  …
    X : C
    S : CategoryTheory.Sieve X
    Y : C
    f : Quiver.Hom Y X
    ⊢ Iff (Eq ((c Y) (CategoryTheory.Sieve.pullback f S)) Top.top) (((c X) S).arro …
  -/
  rw [pb, Sieve.pullback_eq_top_iff_mem]
  /-
    🎉 no goals
  -/


