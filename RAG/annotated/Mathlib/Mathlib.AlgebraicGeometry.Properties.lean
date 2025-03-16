instance : T0Space X :=
  T0Space.of_open_cover fun x => ⟨_, X.affineCover.covers x,
    (X.affineCover.map x).opensRange.2, IsEmbedding.t0Space (Y := PrimeSpectrum _)
    (isAffineOpen_opensRange (X.affineCover.map x)).isoSpec.schemeIsoToHomeo.isEmbedding⟩


instance : QuasiSober X := by
  apply (config := { allowSynthFailures := true })
    quasiSober_of_open_cover (Set.range fun x => Set.range <| (X.affineCover.map x).base)
    /-
      case hS
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (s : ↑(Set.range fun x => Set.range ⇑(X.affineCover.map x).base)), IsOpen ↑s
    -/
  · rintro ⟨_, i, rfl⟩; exact (X.affineCover.map_prop i).base_open.isOpen_range
                        /-
                          🎉 no goals
                        -/
    /-
      case hS'
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (s : ↑(Set.range fun x => Set.range ⇑(X.affineCover.map x).base)), QuasiSo …
    -/
  · rintro ⟨_, i, rfl⟩
    exact @IsOpenEmbedding.quasiSober _ _ _ _ _ (Homeomorph.ofIsEmbedding _
      (X.affineCover.map_prop i).base_open.isEmbedding).symm.isOpenEmbedding
        PrimeSpectrum.quasiSober
    /-
      case hS''
      X : AlgebraicGeometry.Scheme
      ⊢ Eq (Set.range fun x => Set.range ⇑(X.affineCover.map x).base).sUnion Top.top
    -/
  · rw [Set.top_eq_univ, Set.sUnion_range, Set.eq_univ_iff_forall]
    /-
      case hS''
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (x : ↑↑X.toPresheafedSpace), Membership.mem (Set.iUnion fun x => Set.range …
    -/
    intro x; exact ⟨_, ⟨_, rfl⟩, X.affineCover.covers x⟩
             /-
               🎉 no goals
             -/


/-- A scheme `X` is reduced if all `𝒪ₓ(U)` are reduced. -/
class IsReduced : Prop where
  component_reduced : ∀ U, _root_.IsReduced Γ(X, U) := by infer_instance


theorem isReduced_of_isReduced_stalk [∀ x : X, _root_.IsReduced (X.presheaf.stalk x)] :
    IsReduced X := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    ⊢ AlgebraicGeometry.IsReduced X
  -/
  refine ⟨fun U => ⟨fun s hs => ?_⟩⟩
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : IsNilpotent s
    ⊢ Eq s 0
  -/
  apply Presheaf.section_ext X.sheaf U s 0
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : IsNilpotent s
    ⊢ ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), Eq ((X.sheaf.preshe …
  -/
  intro x hx
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : IsNilpotent s
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((X.sheaf.presheaf.germ U x hx) s) ((X.sheaf.presheaf.germ U x hx) 0)
  -/
  show (X.sheaf.presheaf.germ U x hx) s = (X.sheaf.presheaf.germ U x hx) 0
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : IsNilpotent s
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((X.sheaf.presheaf.germ U x hx).hom s) ((X.sheaf.presheaf.germ U x hx).ho …
  -/
  rw [RingHom.map_zero]
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : IsNilpotent s
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((X.sheaf.presheaf.germ U x hx).hom s) 0
  -/
  change X.presheaf.germ U x hx s = 0
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : ∀ (x : ↑↑X.toPresheafedSpace), _root_.IsReduced ↑(X.presheaf.stalk x)
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : IsNilpotent s
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((X.presheaf.germ U x hx).hom s) 0
  -/
  exact (hs.map _).eq_zero
  /-
    🎉 no goals
  -/


instance isReduced_stalk_of_isReduced [IsReduced X] (x : X) :
    _root_.IsReduced (X.presheaf.stalk x) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    ⊢ _root_.IsReduced ↑(X.presheaf.stalk x)
  -/
  constructor
  /-
    case eq_zero
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    ⊢ ∀ (x_1 : ↑(X.presheaf.stalk x)), IsNilpotent x_1 → Eq x_1 0
  -/
  rintro g ⟨n, e⟩
  /-
    case eq_zero.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    g : ↑(X.presheaf.stalk x)
    n : Nat
    e : Eq (HPow.hPow g n) 0
    ⊢ Eq g 0
  -/
  obtain ⟨U, hxU, s, (rfl : (X.presheaf.germ U x hxU) s = g)⟩ := X.presheaf.germ_exist x g
  /-
    case eq_zero.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    n : Nat
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    e : Eq (HPow.hPow ((X.presheaf.germ U x hxU).hom s) n) 0
    ⊢ Eq ((X.presheaf.germ U x hxU).hom s) 0
  -/
  rw [← map_pow, ← map_zero (X.presheaf.germ _ x hxU).hom] at e
  obtain ⟨V, hxV, iU, iV, (e' : (X.presheaf.map iU.op) (s ^ n) = (X.presheaf.map iV.op) 0)⟩ :=
    X.presheaf.germ_eq x hxU hxU _ 0 e
  /-
    case eq_zero.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    n : Nat
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    e : Eq ((X.presheaf.germ U x hxU).hom (HPow.hPow s n)) ((X.presheaf.germ U x h …
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    iU iV : Quiver.Hom V U
    e' : Eq ((X.presheaf.map iU.op).hom (HPow.hPow s n)) ((X.presheaf.map iV.op).h …
    ⊢ Eq ((X.presheaf.germ U x hxU).hom s) 0
  -/
  rw [map_pow, map_zero] at e'
  /-
    case eq_zero.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    n : Nat
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    e : Eq ((X.presheaf.germ U x hxU).hom (HPow.hPow s n)) ((X.presheaf.germ U x h …
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    iU iV : Quiver.Hom V U
    e' : Eq (HPow.hPow ((X.presheaf.map iU.op).hom s) n) 0
    ⊢ Eq ((X.presheaf.germ U x hxU).hom s) 0
  -/
  replace e' := (IsNilpotent.mk _ _ e').eq_zero (R := Γ(X, V))
  /-
    case eq_zero.intro.intro.intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    x : ↑↑X.toPresheafedSpace
    n : Nat
    U : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxU : Membership.mem U x
    s : (CategoryTheory.forget CommRingCat).obj (X.presheaf.obj { unop := U })
    e : Eq ((X.presheaf.germ U x hxU).hom (HPow.hPow s n)) ((X.presheaf.germ U x h …
    V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
    hxV : Membership.mem V x
    iU iV : Quiver.Hom V U
    e' : Eq ((X.presheaf.map iU.op).hom s) 0
    ⊢ Eq ((X.presheaf.germ U x hxU).hom s) 0
  -/
  rw [← X.presheaf.germ_res iU x hxV, CommRingCat.comp_apply, e', map_zero]
  /-
    🎉 no goals
  -/


theorem isReduced_of_isOpenImmersion {X Y : Scheme} (f : X ⟶ Y) [H : IsOpenImmersion f]
    [IsReduced Y] : IsReduced X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsReduced Y
    ⊢ AlgebraicGeometry.IsReduced X
  -/
  constructor
  /-
    case component_reduced
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsReduced Y
    ⊢ autoParam (∀ (U : X.Opens), _root_.IsReduced ↑(X.presheaf.obj { unop := U }) …
  -/
  intro U
  have : U = f ⁻¹ᵁ f ''ᵁ U := by
    ext1; exact (Set.preimage_image_eq _ H.base_open.injective).symm
  /-
    case component_reduced
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsReduced Y
    U : X.Opens
    this : Eq U ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Schem …
    ⊢ _root_.IsReduced ↑(X.presheaf.obj { unop := U })
  -/
  rw [this]
  exact isReduced_of_injective (inv <| f.app (f ''ᵁ U)).hom
    (asIso <| f.app (f ''ᵁ U) : Γ(Y, f ''ᵁ U) ≅ _).symm.commRingCatIsoToRingEquiv.injective


instance {R : CommRingCat.{u}} [H : _root_.IsReduced R] : IsReduced (Spec R) := by
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    H : _root_.IsReduced ↑R
    ⊢ AlgebraicGeometry.IsReduced (AlgebraicGeometry.Spec R)
  -/
  apply (config := { allowSynthFailures := true }) isReduced_of_isReduced_stalk
  /-
    case inst
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    H : _root_.IsReduced ↑R
    ⊢ ∀ (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), _root_.IsReduced ↑(( …
  -/
  intro x; dsimp
  have : _root_.IsReduced (CommRingCat.of <| Localization.AtPrime (PrimeSpectrum.asIdeal x)) := by
    dsimp; infer_instance
  exact isReduced_of_injective (StructureSheaf.stalkIso R x).hom.hom
    (StructureSheaf.stalkIso R x).commRingCatIsoToRingEquiv.injective


theorem affine_isReduced_iff (R : CommRingCat) :
    IsReduced (Spec R) ↔ _root_.IsReduced R := by
  /-
    R : CommRingCat
    ⊢ Iff (AlgebraicGeometry.IsReduced (AlgebraicGeometry.Spec R)) (_root_.IsReduc …
  -/
  refine ⟨?_, fun h => inferInstance⟩
  /-
    R : CommRingCat
    ⊢ AlgebraicGeometry.IsReduced (AlgebraicGeometry.Spec R) → _root_.IsReduced ↑R
  -/
  intro h
  exact isReduced_of_injective (Scheme.ΓSpecIso R).inv.hom
    (Scheme.ΓSpecIso R).symm.commRingCatIsoToRingEquiv.injective


theorem isReduced_of_isAffine_isReduced [IsAffine X] [_root_.IsReduced Γ(X, ⊤)] :
    IsReduced X :=
  isReduced_of_isOpenImmersion X.isoSpec.hom


/-- To show that a statement `P` holds for all open subsets of all schemes, it suffices to show that
1. In any scheme `X`, if `P` holds for an open cover of `U`, then `P` holds for `U`.
2. For an open immerison `f : X ⟶ Y`, if `P` holds for the entire space of `X`, then `P` holds for
  the image of `f`.
3. `P` holds for the entire space of an affine scheme.
-/
@[elab_as_elim]
theorem reduce_to_affine_global (P : ∀ {X : Scheme} (_ : X.Opens), Prop)
    {X : Scheme} (U : X.Opens)
    (h₁ : ∀ (X : Scheme) (U : X.Opens),
      (∀ x : U, ∃ (V : _) (_ : x.1 ∈ V) (_ : V ⟶ U), P V) → P U)
    (h₂ : ∀ (X Y) (f : X ⟶ Y) [IsOpenImmersion f],
      ∃ (U : X.Opens) (V : Y.Opens), U = ⊤ ∧ V = f.opensRange ∧ (P U → P V))
    (h₃ : ∀ R : CommRingCat, P (X := Spec R) ⊤) : P U := by
  /-
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    ⊢ P U
  -/
  apply h₁
  /-
    case a
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    ⊢ ∀ (x : Subtype fun x => Membership.mem U x), Exists fun V => Exists fun x => …
  -/
  intro x
  obtain ⟨_, ⟨j, rfl⟩, hx, i⟩ :=
    X.affineBasisCover_is_basis.exists_subset_of_mem_open (SetLike.mem_coe.2 x.prop) U.isOpen
  /-
    case a.intro.intro.intro.intro
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    x : Subtype fun x => Membership.mem U x
    j : X.affineBasisCover.J
    hx : Membership.mem (Set.range ⇑(X.affineBasisCover.map j).base) ↑x
    i : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map j).base) ↑U
    ⊢ Exists fun V => Exists fun x => Exists fun x => P V
  -/
  let U' : Opens _ := ⟨_, (X.affineBasisCover.map_prop j).base_open.isOpen_range⟩
  /-
    case a.intro.intro.intro.intro
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    x : Subtype fun x => Membership.mem U x
    j : X.affineBasisCover.J
    hx : Membership.mem (Set.range ⇑(X.affineBasisCover.map j).base) ↑x
    i : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map j).base) ↑U
    U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := { carrier := Set.range ⇑( …
    ⊢ Exists fun V => Exists fun x => Exists fun x => P V
  -/
  let i' : U' ⟶ U := homOfLE i
  /-
    case a.intro.intro.intro.intro
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    x : Subtype fun x => Membership.mem U x
    j : X.affineBasisCover.J
    hx : Membership.mem (Set.range ⇑(X.affineBasisCover.map j).base) ↑x
    i : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map j).base) ↑U
    U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := { carrier := Set.range ⇑( …
    i' : Quiver.Hom U' U := CategoryTheory.homOfLE i
    ⊢ Exists fun V => Exists fun x => Exists fun x => P V
  -/
  refine ⟨U', hx, i', ?_⟩
  /-
    case a.intro.intro.intro.intro
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    x : Subtype fun x => Membership.mem U x
    j : X.affineBasisCover.J
    hx : Membership.mem (Set.range ⇑(X.affineBasisCover.map j).base) ↑x
    i : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map j).base) ↑U
    U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := { carrier := Set.range ⇑( …
    i' : Quiver.Hom U' U := CategoryTheory.homOfLE i
    ⊢ P U'
  -/
  obtain ⟨_, _, rfl, rfl, h₂'⟩ := h₂ _ _ (X.affineBasisCover.map j)
  /-
    case a.intro.intro.intro.intro.intro.intro.intro.intro
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    x : Subtype fun x => Membership.mem U x
    j : X.affineBasisCover.J
    hx : Membership.mem (Set.range ⇑(X.affineBasisCover.map j).base) ↑x
    i : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map j).base) ↑U
    U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := { carrier := Set.range ⇑( …
    i' : Quiver.Hom U' U := CategoryTheory.homOfLE i
    h₂' : P Top.top → P (AlgebraicGeometry.Scheme.Hom.opensRange (X.affineBasisCov …
    ⊢ P U'
  -/
  apply h₂'
  /-
    case a.intro.intro.intro.intro.intro.intro.intro.intro
    P : {X : AlgebraicGeometry.Scheme} → X.Opens → Prop
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    h₁ : ∀ (X : AlgebraicGeometry.Scheme) (U : X.Opens), (∀ (x : Subtype fun x =>  …
    h₂ : ∀ (X Y : AlgebraicGeometry.Scheme) (f : Quiver.Hom X Y) [inst : Algebraic …
    h₃ : ∀ (R : CommRingCat), P Top.top
    x : Subtype fun x => Membership.mem U x
    j : X.affineBasisCover.J
    hx : Membership.mem (Set.range ⇑(X.affineBasisCover.map j).base) ↑x
    i : HasSubset.Subset (Set.range ⇑(X.affineBasisCover.map j).base) ↑U
    U' : TopologicalSpace.Opens ↑↑X.toPresheafedSpace := { carrier := Set.range ⇑( …
    i' : Quiver.Hom U' U := CategoryTheory.homOfLE i
    h₂' : P Top.top → P (AlgebraicGeometry.Scheme.Hom.opensRange (X.affineBasisCov …
    ⊢ P Top.top
  -/
  apply h₃
  /-
    🎉 no goals
  -/


theorem reduce_to_affine_nbhd (P : ∀ (X : Scheme) (_ : X), Prop)
    (h₁ : ∀ R x, P (Spec R) x)
    (h₂ : ∀ {X Y} (f : X ⟶ Y) [IsOpenImmersion f] (x : X), P X x → P Y (f.base x)) :
    ∀ (X : Scheme) (x : X), P X x := by
  /-
    P : (X : AlgebraicGeometry.Scheme) → ↑↑X.toPresheafedSpace → Prop
    h₁ : ∀ (R : CommRingCat) (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), …
    h₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebraic …
    ⊢ ∀ (X : AlgebraicGeometry.Scheme) (x : ↑↑X.toPresheafedSpace), P X x
  -/
  intro X x
  /-
    P : (X : AlgebraicGeometry.Scheme) → ↑↑X.toPresheafedSpace → Prop
    h₁ : ∀ (R : CommRingCat) (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), …
    h₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebraic …
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    ⊢ P X x
  -/
  obtain ⟨y, e⟩ := X.affineCover.covers x
  /-
    case intro
    P : (X : AlgebraicGeometry.Scheme) → ↑↑X.toPresheafedSpace → Prop
    h₁ : ∀ (R : CommRingCat) (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), …
    h₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebraic …
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(X.affineCover.obj (X.affineCover.f x)).toPresheafedSpace
    e : Eq ((X.affineCover.map (X.affineCover.f x)).base y) x
    ⊢ P X x
  -/
  convert h₂ (X.affineCover.map (X.affineCover.f x)) y _
    /-
      case h.e'_2
      P : (X : AlgebraicGeometry.Scheme) → ↑↑X.toPresheafedSpace → Prop
      h₁ : ∀ (R : CommRingCat) (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), …
      h₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebraic …
      X : AlgebraicGeometry.Scheme
      x : ↑↑X.toPresheafedSpace
      y : ↑↑(X.affineCover.obj (X.affineCover.f x)).toPresheafedSpace
      e : Eq ((X.affineCover.map (X.affineCover.f x)).base y) x
      ⊢ Eq x ((X.affineCover.map (X.affineCover.f x)).base y)
    -/
  · rw [e]
    /-
      🎉 no goals
    -/
  /-
    case intro
    P : (X : AlgebraicGeometry.Scheme) → ↑↑X.toPresheafedSpace → Prop
    h₁ : ∀ (R : CommRingCat) (x : ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace), …
    h₂ : ∀ {X Y : AlgebraicGeometry.Scheme} (f : Quiver.Hom X Y) [inst : Algebraic …
    X : AlgebraicGeometry.Scheme
    x : ↑↑X.toPresheafedSpace
    y : ↑↑(X.affineCover.obj (X.affineCover.f x)).toPresheafedSpace
    e : Eq ((X.affineCover.map (X.affineCover.f x)).base y) x
    ⊢ P (X.affineCover.obj (X.affineCover.f x)) y
  -/
  apply h₁
  /-
    🎉 no goals
  -/


theorem eq_zero_of_basicOpen_eq_bot {X : Scheme} [hX : IsReduced X] {U : X.Opens}
    (s : Γ(X, U)) (hs : X.basicOpen s = ⊥) : s = 0 := by
  /-
    X : AlgebraicGeometry.Scheme
    hX : AlgebraicGeometry.IsReduced X
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : Eq (X.basicOpen s) Bot.bot
    ⊢ Eq s 0
  -/
  apply TopCat.Presheaf.section_ext X.sheaf U
  /-
    case h
    X : AlgebraicGeometry.Scheme
    hX : AlgebraicGeometry.IsReduced X
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : Eq (X.basicOpen s) Bot.bot
    ⊢ ∀ (x : ↑↑X.toPresheafedSpace) (hx : Membership.mem U x), Eq ((X.sheaf.preshe …
  -/
  intro x hx
  /-
    case h
    X : AlgebraicGeometry.Scheme
    hX : AlgebraicGeometry.IsReduced X
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : Eq (X.basicOpen s) Bot.bot
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((X.sheaf.presheaf.germ U x hx) s) ((X.sheaf.presheaf.germ U x hx) 0)
  -/
  show (X.sheaf.presheaf.germ U x hx) s = (X.sheaf.presheaf.germ U x hx) 0
  /-
    case h
    X : AlgebraicGeometry.Scheme
    hX : AlgebraicGeometry.IsReduced X
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    hs : Eq (X.basicOpen s) Bot.bot
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U x
    ⊢ Eq ((X.sheaf.presheaf.germ U x hx).hom s) ((X.sheaf.presheaf.germ U x hx).ho …
  -/
  rw [RingHom.map_zero]
  induction U using reduce_to_affine_global generalizing hX with
  | h₁ X U H =>
    obtain ⟨V, hx, i, H⟩ := H ⟨x, hx⟩
    specialize H (X.presheaf.map i.op s)
    rw [Scheme.basicOpen_res, hs] at H
    specialize H (inf_bot_eq _) x hx
    -- This seems to be related to a mismatch of `X.sheaf.presheaf` and `X.presheaf` in `H`
    rw [← CommRingCat.germ_res_apply X.sheaf.presheaf i x hx s]
    exact H
  | h₂ X Y f =>
    refine ⟨f ⁻¹ᵁ f.opensRange, f.opensRange, by ext1; simp, rfl, ?_⟩
    rintro H hX s hs _ ⟨x, rfl⟩
    haveI := isReduced_of_isOpenImmersion f
    specialize H (f.app _ s) _ x ⟨x, rfl⟩
    · rw [← Scheme.preimage_basicOpen, hs]; ext1; simp [Opens.map]
    · have H : (X.presheaf.germ _ x _).hom _ = 0 := H
      rw [← Scheme.stalkMap_germ_apply f ⟨_, _⟩ x] at H
      apply_fun inv <| f.stalkMap x at H
      rw [← CommRingCat.comp_apply, CategoryTheory.IsIso.hom_inv_id, map_zero] at H
      exact H
  | h₃ R =>
    rw [basicOpen_eq_of_affine', PrimeSpectrum.basicOpen_eq_bot_iff] at hs
    replace hs := (hs.map (Scheme.ΓSpecIso R).inv.hom).eq_zero
    rw [← CommRingCat.comp_apply, Iso.hom_inv_id, CommRingCat.id_apply] at hs
    rw [hs, map_zero]


@[simp]
theorem basicOpen_eq_bot_iff {X : Scheme} [IsReduced X] {U : X.Opens}
    (s : Γ(X, U)) : X.basicOpen s = ⊥ ↔ s = 0 := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    ⊢ Iff (Eq (X.basicOpen s) Bot.bot) (Eq s 0)
  -/
  refine ⟨eq_zero_of_basicOpen_eq_bot s, ?_⟩
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    U : X.Opens
    s : ↑(X.presheaf.obj { unop := U })
    ⊢ Eq s 0 → Eq (X.basicOpen s) Bot.bot
  -/
  rintro rfl
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    U : X.Opens
    ⊢ Eq (X.basicOpen 0) Bot.bot
  -/
  simp
  /-
    🎉 no goals
  -/


/-- A scheme `X` is integral if its is nonempty,
and `𝒪ₓ(U)` is an integral domain for each `U ≠ ∅`. -/
class IsIntegral : Prop where
  nonempty : Nonempty X := by infer_instance
  component_integral : ∀ (U : X.Opens) [Nonempty U], IsDomain Γ(X, U) := by infer_instance


instance [IsIntegral X] : IsDomain Γ(X, ⊤) :=
  @IsIntegral.component_integral _ _ _ ⟨Nonempty.some inferInstance, trivial⟩


instance (priority := 900) isReduced_of_isIntegral [IsIntegral X] : IsReduced X := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    ⊢ AlgebraicGeometry.IsReduced X
  -/
  constructor
  /-
    case component_reduced
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    ⊢ autoParam (∀ (U : X.Opens), _root_.IsReduced ↑(X.presheaf.obj { unop := U }) …
  -/
  intro U
  /-
    case component_reduced
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U : X.Opens
    ⊢ _root_.IsReduced ↑(X.presheaf.obj { unop := U })
  -/
  rcases U.1.eq_empty_or_nonempty with h | h
    /-
      case component_reduced.inl
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      U : X.Opens
      h : Eq U.carrier EmptyCollection.emptyCollection
      ⊢ _root_.IsReduced ↑(X.presheaf.obj { unop := U })
    -/
  · have : U = ⊥ := SetLike.ext' h
    haveI : Subsingleton Γ(X, U) :=
      CommRingCat.subsingleton_of_isTerminal (X.sheaf.isTerminalOfEqEmpty this)
    /-
      case component_reduced.inl
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      U : X.Opens
      h : Eq U.carrier EmptyCollection.emptyCollection
      this✝ : Eq U Bot.bot
      this : Subsingleton ↑(X.presheaf.obj { unop := U })
      ⊢ _root_.IsReduced ↑(X.presheaf.obj { unop := U })
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case component_reduced.inr
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      U : X.Opens
      h : U.carrier.Nonempty
      ⊢ _root_.IsReduced ↑(X.presheaf.obj { unop := U })
    -/
  · haveI : Nonempty U := by simpa
    /-
      case component_reduced.inr
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      U : X.Opens
      h : U.carrier.Nonempty
      this : Nonempty ↑↑(↑U).toPresheafedSpace
      ⊢ _root_.IsReduced ↑(X.presheaf.obj { unop := U })
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance Scheme.component_nontrivial (X : Scheme.{u}) (U : X.Opens) [Nonempty U] :
    Nontrivial Γ(X, U) :=
  LocallyRingedSpace.component_nontrivial (hU := ‹_›)


instance irreducibleSpace_of_isIntegral [IsIntegral X] : IrreducibleSpace X := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    ⊢ IrreducibleSpace ↑↑X.toPresheafedSpace
  -/
  by_contra H
  replace H : ¬IsPreirreducible (⊤ : Set X) := fun h =>
    H { toPreirreducibleSpace := ⟨h⟩
        toNonempty := inferInstance }
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    H : Not (IsPreirreducible Top.top)
    ⊢ False
  -/
  simp_rw [isPreirreducible_iff_isClosed_union_isClosed, not_forall, not_or] at H
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    H : Exists fun x => Exists fun x_1 => Exists fun h => Exists fun h => Exists f …
    ⊢ False
  -/
  rcases H with ⟨S, T, hS, hT, h₁, h₂, h₃⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    S T : Set ↑↑X.toPresheafedSpace
    hS : IsClosed S
    hT : IsClosed T
    h₁ : HasSubset.Subset Top.top (Union.union S T)
    h₂ : Not (HasSubset.Subset Top.top S)
    h₃ : Not (HasSubset.Subset Top.top T)
    ⊢ False
  -/
  erw [not_forall] at h₂ h₃
  /-
    case intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    S T : Set ↑↑X.toPresheafedSpace
    hS : IsClosed S
    hT : IsClosed T
    h₁ : HasSubset.Subset Top.top (Union.union S T)
    h₂ : Exists fun x => Not (Membership.mem Top.top x → Membership.mem S x)
    h₃ : Exists fun x => Not (Membership.mem Top.top x → Membership.mem T x)
    ⊢ False
  -/
  simp_rw [not_forall] at h₂ h₃
  /-
    case intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    S T : Set ↑↑X.toPresheafedSpace
    hS : IsClosed S
    hT : IsClosed T
    h₁ : HasSubset.Subset Top.top (Union.union S T)
    h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
    h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
    ⊢ False
  -/
  haveI : Nonempty (⟨Sᶜ, hS.1⟩ : X.Opens) := ⟨⟨_, h₂.choose_spec.choose_spec⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    S T : Set ↑↑X.toPresheafedSpace
    hS : IsClosed S
    hT : IsClosed T
    h₁ : HasSubset.Subset Top.top (Union.union S T)
    h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
    h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
    this : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPresheaf …
    ⊢ False
  -/
  haveI : Nonempty (⟨Tᶜ, hT.1⟩ : X.Opens) := ⟨⟨_, h₃.choose_spec.choose_spec⟩⟩
  haveI : Nonempty (⟨Sᶜ, hS.1⟩ ⊔ ⟨Tᶜ, hT.1⟩ : X.Opens) :=
    ⟨⟨_, Or.inl h₂.choose_spec.choose_spec⟩⟩
  let e : Γ(X, _) ≅ CommRingCat.of _ :=
    (X.sheaf.isProductOfDisjoint ⟨_, hS.1⟩ ⟨_, hT.1⟩ ?_).conePointUniqueUpToIso
      (CommRingCat.prodFanIsLimit _ _)
  · have : IsDomain (Γ(X, ⟨Sᶜ, hS.1⟩) × Γ(X, ⟨Tᶜ, hT.1⟩)) :=
      e.symm.commRingCatIsoToRingEquiv.toMulEquiv.isDomain _
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      S T : Set ↑↑X.toPresheafedSpace
      hS : IsClosed S
      hT : IsClosed T
      h₁ : HasSubset.Subset Top.top (Union.union S T)
      h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
      h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
      this✝² : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
      this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshe …
      this✝ : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } { …
      e : CategoryTheory.Iso (X.presheaf.obj { unop := Max.max { carrier := HasCompl …
      this : IsDomain (Prod ↑(X.presheaf.obj { unop := { carrier := HasCompl.compl S …
      ⊢ False
    -/
    exact false_of_nontrivial_of_product_domain Γ(X, ⟨Sᶜ, hS.1⟩) Γ(X, ⟨Tᶜ, hT.1⟩)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      S T : Set ↑↑X.toPresheafedSpace
      hS : IsClosed S
      hT : IsClosed T
      h₁ : HasSubset.Subset Top.top (Union.union S T)
      h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
      h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
      this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
      this✝ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshea …
      this : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } {  …
      ⊢ Eq (Min.min { carrier := HasCompl.compl S, is_open' := ⋯ } { carrier := HasC …
    -/
  · ext x
    /-
      case intro.intro.intro.intro.intro.intro.refine_1.h.h
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsIntegral X
      S T : Set ↑↑X.toPresheafedSpace
      hS : IsClosed S
      hT : IsClosed T
      h₁ : HasSubset.Subset Top.top (Union.union S T)
      h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
      h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
      this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
      this✝ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshea …
      this : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } {  …
      x : ↑↑X.toPresheafedSpace
      ⊢ Iff (Membership.mem (↑(Min.min { carrier := HasCompl.compl S, is_open' := ⋯  …
    -/
    constructor
      /-
        case intro.intro.intro.intro.intro.intro.refine_1.h.h.mp
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsIntegral X
        S T : Set ↑↑X.toPresheafedSpace
        hS : IsClosed S
        hT : IsClosed T
        h₁ : HasSubset.Subset Top.top (Union.union S T)
        h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
        h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
        this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
        this✝ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshea …
        this : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } {  …
        x : ↑↑X.toPresheafedSpace
        ⊢ Membership.mem (↑(Min.min { carrier := HasCompl.compl S, is_open' := ⋯ } { c …
      -/
    · rintro ⟨hS, hT⟩
      /-
        case intro.intro.intro.intro.intro.intro.refine_1.h.h.mp.intro
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsIntegral X
        S T : Set ↑↑X.toPresheafedSpace
        hS✝ : IsClosed S
        hT✝ : IsClosed T
        h₁ : HasSubset.Subset Top.top (Union.union S T)
        h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
        h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
        this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
        this✝ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshea …
        this : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } {  …
        x : ↑↑X.toPresheafedSpace
        hS : Membership.mem (↑{ carrier := HasCompl.compl S, is_open' := ⋯ }) x
        hT : Membership.mem (↑{ carrier := HasCompl.compl T, is_open' := ⋯ }) x
        ⊢ Membership.mem (↑Bot.bot) x
      -/
      cases' h₁ (show x ∈ ⊤ by trivial) with h h
      /-
        case intro.intro.intro.intro.intro.intro.refine_1.h.h.mp.intro.inl
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsIntegral X
        S T : Set ↑↑X.toPresheafedSpace
        hS✝ : IsClosed S
        hT✝ : IsClosed T
        h₁ : HasSubset.Subset Top.top (Union.union S T)
        h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
        h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
        this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
        this✝ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshea …
        this : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } {  …
        x : ↑↑X.toPresheafedSpace
        hS : Membership.mem (↑{ carrier := HasCompl.compl S, is_open' := ⋯ }) x
        hT : Membership.mem (↑{ carrier := HasCompl.compl T, is_open' := ⋯ }) x
        h : Membership.mem S x
        ⊢ Membership.mem (↑Bot.bot) x
      -/
      exacts [hS h, hT h]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_1.h.h.mpr
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsIntegral X
        S T : Set ↑↑X.toPresheafedSpace
        hS : IsClosed S
        hT : IsClosed T
        h₁ : HasSubset.Subset Top.top (Union.union S T)
        h₂ : Exists fun x => Exists fun x_1 => Not (Membership.mem S x)
        h₃ : Exists fun x => Exists fun x_1 => Not (Membership.mem T x)
        this✝¹ : Nonempty ↑↑(↑{ carrier := HasCompl.compl S, is_open' := ⋯ }).toPreshe …
        this✝ : Nonempty ↑↑(↑{ carrier := HasCompl.compl T, is_open' := ⋯ }).toPreshea …
        this : Nonempty ↑↑(↑(Max.max { carrier := HasCompl.compl S, is_open' := ⋯ } {  …
        x : ↑↑X.toPresheafedSpace
        ⊢ Membership.mem (↑Bot.bot) x → Membership.mem (↑(Min.min { carrier := HasComp …
      -/
    · simp
      /-
        🎉 no goals
      -/


theorem isIntegral_of_irreducibleSpace_of_isReduced [IsReduced X] [H : IrreducibleSpace X] :
    IsIntegral X := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    H : IrreducibleSpace ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsIntegral X
  -/
  constructor; · infer_instance
                 /-
                   🎉 no goals
                 -/
  /-
    case component_integral
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    H : IrreducibleSpace ↑↑X.toPresheafedSpace
    ⊢ autoParam (∀ (U : X.Opens) [inst : Nonempty ↑↑(↑U).toPresheafedSpace], IsDom …
  -/
  intro U hU
  /-
    case component_integral
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    H : IrreducibleSpace ↑↑X.toPresheafedSpace
    U : X.Opens
    hU : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ IsDomain ↑(X.presheaf.obj { unop := U })
  -/
  haveI := (@LocallyRingedSpace.component_nontrivial X.toLocallyRingedSpace U hU).1
  have : NoZeroDivisors
      (X.toLocallyRingedSpace.toSheafedSpace.toPresheafedSpace.presheaf.obj (op U)) := by
    refine ⟨fun {a b} e => ?_⟩
    simp_rw [← basicOpen_eq_bot_iff, ← Opens.not_nonempty_iff_eq_bot]
    by_contra! h
    obtain ⟨x, ⟨hxU, hx₁⟩, _, hx₂⟩ :=
      nonempty_preirreducible_inter (X.basicOpen a).2 (X.basicOpen b).2 h.1 h.2
    replace e := congr_arg (X.presheaf.germ U x hxU) e
    rw [RingHom.map_mul, RingHom.map_zero] at e
    refine zero_ne_one' (X.presheaf.stalk x) (isUnit_zero_iff.1 ?_)
    convert hx₁.mul hx₂
    exact e.symm
  /-
    case component_integral
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsReduced X
    H : IrreducibleSpace ↑↑X.toPresheafedSpace
    U : X.Opens
    hU : Nonempty ↑↑(↑U).toPresheafedSpace
    this✝ : Exists fun x => Exists fun y => Ne x y
    this : NoZeroDivisors ↑(X.presheaf.obj { unop := U })
    ⊢ IsDomain ↑(X.presheaf.obj { unop := U })
  -/
  exact NoZeroDivisors.to_isDomain _
  /-
    🎉 no goals
  -/


theorem isIntegral_iff_irreducibleSpace_and_isReduced :
    IsIntegral X ↔ IrreducibleSpace X ∧ IsReduced X :=
  ⟨fun _ => ⟨inferInstance, inferInstance⟩, fun ⟨_, _⟩ =>
    isIntegral_of_irreducibleSpace_of_isReduced X⟩


theorem isIntegral_of_isOpenImmersion {X Y : Scheme} (f : X ⟶ Y) [H : IsOpenImmersion f]
    [IsIntegral Y] [Nonempty X] : IsIntegral X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝¹ : AlgebraicGeometry.IsIntegral Y
    inst✝ : Nonempty ↑↑X.toPresheafedSpace
    ⊢ AlgebraicGeometry.IsIntegral X
  -/
  constructor; · infer_instance
                 /-
                   🎉 no goals
                 -/
  /-
    case component_integral
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝¹ : AlgebraicGeometry.IsIntegral Y
    inst✝ : Nonempty ↑↑X.toPresheafedSpace
    ⊢ autoParam (∀ (U : X.Opens) [inst : Nonempty ↑↑(↑U).toPresheafedSpace], IsDom …
  -/
  intro U hU
  /-
    case component_integral
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝¹ : AlgebraicGeometry.IsIntegral Y
    inst✝ : Nonempty ↑↑X.toPresheafedSpace
    U : X.Opens
    hU : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ IsDomain ↑(X.presheaf.obj { unop := U })
  -/
  have : U = f ⁻¹ᵁ f ''ᵁ U := by ext1; exact (Set.preimage_image_eq _ H.base_open.injective).symm
  /-
    case component_integral
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    H : AlgebraicGeometry.IsOpenImmersion f
    inst✝¹ : AlgebraicGeometry.IsIntegral Y
    inst✝ : Nonempty ↑↑X.toPresheafedSpace
    U : X.Opens
    hU : Nonempty ↑↑(↑U).toPresheafedSpace
    this : Eq U ((TopologicalSpace.Opens.map f.base).obj ((AlgebraicGeometry.Schem …
    ⊢ IsDomain ↑(X.presheaf.obj { unop := U })
  -/
  rw [this]
  have : IsDomain Γ(Y, f ''ᵁ U) := by
    apply (config := { allowSynthFailures := true }) IsIntegral.component_integral
    exact ⟨⟨_, _, hU.some.prop, rfl⟩⟩
  exact (asIso <| f.app (f ''ᵁ U) :
    Γ(Y, f ''ᵁ U) ≅ _).symm.commRingCatIsoToRingEquiv.toMulEquiv.isDomain _


instance {R : CommRingCat} [IsDomain R] : IrreducibleSpace (Spec R) := by
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsDomain ↑R
    ⊢ IrreducibleSpace ↑↑(AlgebraicGeometry.Spec R).toPresheafedSpace
  -/
  convert PrimeSpectrum.irreducibleSpace (R := R)
  /-
    🎉 no goals
  -/


instance {R : CommRingCat} [IsDomain R] : IsIntegral (Spec R) :=
  isIntegral_of_irreducibleSpace_of_isReduced _


theorem affine_isIntegral_iff (R : CommRingCat) :
    IsIntegral (Spec R) ↔ IsDomain R :=
  ⟨fun _ => MulEquiv.isDomain Γ(Spec R, ⊤)
    (Scheme.ΓSpecIso R).symm.commRingCatIsoToRingEquiv.toMulEquiv, fun _ => inferInstance⟩


theorem isIntegral_of_isAffine_of_isDomain [IsAffine X] [Nonempty X] [IsDomain Γ(X, ⊤)] :
    IsIntegral X :=
  isIntegral_of_isOpenImmersion X.isoSpec.hom


theorem map_injective_of_isIntegral [IsIntegral X] {U V : X.Opens} (i : U ⟶ V)
    [H : Nonempty U] : Function.Injective (X.presheaf.map i.op) := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ Function.Injective ⇑(X.presheaf.map i.op).hom
  -/
  rw [injective_iff_map_eq_zero]
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    ⊢ ∀ (a : ↑(X.presheaf.obj { unop := V })), Eq ((X.presheaf.map i.op).hom a) 0  …
  -/
  intro x hx
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    hx : Eq ((X.presheaf.map i.op).hom x) 0
    ⊢ Eq x 0
  -/
  rw [← basicOpen_eq_bot_iff] at hx ⊢
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    hx : Eq (X.basicOpen ((X.presheaf.map i.op).hom x)) Bot.bot
    ⊢ Eq (X.basicOpen x) Bot.bot
  -/
  rw [Scheme.basicOpen_res] at hx
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    hx : Eq (Min.min U (X.basicOpen x)) Bot.bot
    ⊢ Eq (X.basicOpen x) Bot.bot
  -/
  revert hx
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    ⊢ Eq (Min.min U (X.basicOpen x)) Bot.bot → Eq (X.basicOpen x) Bot.bot
  -/
  contrapose!
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    ⊢ Ne (X.basicOpen x) Bot.bot → Ne (Min.min U (X.basicOpen x)) Bot.bot
  -/
  simp_rw [Ne, ← Opens.not_nonempty_iff_eq_bot, Classical.not_not]
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    ⊢ (↑(X.basicOpen x)).Nonempty → (↑(Min.min U (X.basicOpen x))).Nonempty
  -/
  apply nonempty_preirreducible_inter U.isOpen (RingedSpace.basicOpen _ _).isOpen
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsIntegral X
    U V : X.Opens
    i : Quiver.Hom U V
    H : Nonempty ↑↑(↑U).toPresheafedSpace
    x : ↑(X.presheaf.obj { unop := V })
    ⊢ (↑U).Nonempty
  -/
  simpa using H
  /-
    🎉 no goals
  -/


