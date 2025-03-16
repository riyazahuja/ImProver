/-- The categorical subobjects of a module `M` are in one-to-one correspondence with its
    submodules. -/
noncomputable def subobjectModule : Subobject M ≃o Submodule R M :=
  OrderIso.symm
    { invFun := fun S => LinearMap.range S.arrow.hom
      toFun := fun N => Subobject.mk (ofHom N.subtype)
      right_inv := fun S => Eq.symm (by
        /-
          R : Type u
          inst✝ : Ring R
          M : ModuleCat R
          S : CategoryTheory.Subobject M
          ⊢ Eq S ((fun N => CategoryTheory.Subobject.mk (ModuleCat.ofHom N.subtype)) ((f …
        -/
        fapply eq_mk_of_comm
          /-
            case i
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S : CategoryTheory.Subobject M
            ⊢ CategoryTheory.Iso (CategoryTheory.Subobject.underlying.obj S) (ModuleCat.of …
          -/
        · apply LinearEquiv.toModuleIso
          apply LinearEquiv.ofBijective (LinearMap.codRestrict
            (LinearMap.range S.arrow.hom) S.arrow.hom _)
          /-
            case i.e
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S : CategoryTheory.Subobject M
            ⊢ Function.Bijective ⇑(LinearMap.codRestrict (LinearMap.range S.arrow.hom) S.a …
          -/
          constructor
            /-
              case i.e.left
              R : Type u
              inst✝ : Ring R
              M : ModuleCat R
              S : CategoryTheory.Subobject M
              ⊢ Function.Injective ⇑(LinearMap.codRestrict (LinearMap.range S.arrow.hom) S.a …
            -/
          · simp [← LinearMap.ker_eq_bot, LinearMap.ker_codRestrict]
            /-
              case i.e.left
              R : Type u
              inst✝ : Ring R
              M : ModuleCat R
              S : CategoryTheory.Subobject M
              ⊢ Eq (LinearMap.ker S.arrow.hom) Bot.bot
            -/
            rw [ker_eq_bot_of_mono]
            /-
              🎉 no goals
            -/
          · rw [← LinearMap.range_eq_top, LinearMap.range_codRestrict,
              Submodule.comap_subtype_self]
            /-
              case i.e.right.hf
              R : Type u
              inst✝ : Ring R
              M : ModuleCat R
              S : CategoryTheory.Subobject M
              ⊢ ∀ (c : ↑(CategoryTheory.Subobject.underlying.obj S)), Membership.mem (Linear …
            -/
            exact LinearMap.mem_range_self _
            /-
              🎉 no goals
            -/
          /-
            case w
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S : CategoryTheory.Subobject M
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (LinearEquiv.ofBijective (LinearMap.c …
          -/
        · ext x
          /-
            case w.hf.h
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S : CategoryTheory.Subobject M
            x : ↑(CategoryTheory.Subobject.underlying.obj S)
            ⊢ Eq ((CategoryTheory.CategoryStruct.comp (LinearEquiv.ofBijective (LinearMap. …
          -/
          /-
            case h.e'_2
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            N : Submodule R ↑M
            this : Eq (CategoryTheory.Subobject.underlyingIso (ModuleCat.ofHom N.subtype)) …
            ⊢ Eq ((fun S => LinearMap.range S.arrow.hom) ((fun N => CategoryTheory.Subobje …
          -/
          rfl)
          /-
            🎉 no goals
          -/
          /-
            case h.e'_3
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            N : Submodule R ↑M
            ⊢ Eq N (LinearMap.range (ModuleCat.ofHom N.subtype).hom)
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
      left_inv := fun N => by
        convert congr_arg LinearMap.range (ModuleCat.hom_ext_iff.mp
            (underlyingIso_arrow (ofHom N.subtype))) using 1
        · have :
            -- Porting note: added the `.toLinearEquiv.toLinearMap`
            (underlyingIso (ofHom N.subtype)).inv =
              ofHom (underlyingIso (ofHom N.subtype)).symm.toLinearEquiv.toLinearMap := by
              ext x
              rfl
          rw [this, hom_comp, LinearEquiv.range_comp]
        · exact (Submodule.range_subtype _).symm
      map_rel_iff' := fun {S T} => by
        /-
          R : Type u
          inst✝ : Ring R
          M : ModuleCat R
          S T : Submodule R ↑M
          ⊢ Iff (LE.le ({ toFun := fun N => CategoryTheory.Subobject.mk (ModuleCat.ofHom …
        -/
        refine ⟨fun h => ?_, fun h => mk_le_mk_of_comm (↟(Submodule.inclusion h)) rfl⟩
        /-
          R : Type u
          inst✝ : Ring R
          M : ModuleCat R
          S T : Submodule R ↑M
          h : LE.le ({ toFun := fun N => CategoryTheory.Subobject.mk (ModuleCat.ofHom N. …
          ⊢ LE.le S T
        -/
        convert LinearMap.range_comp_le_range (ofMkLEMk _ _ h).hom (ofHom T.subtype).hom
          /-
            case h.e'_3
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S T : Submodule R ↑M
            h : LE.le ({ toFun := fun N => CategoryTheory.Subobject.mk (ModuleCat.ofHom N. …
            ⊢ Eq S (LinearMap.range ((ModuleCat.ofHom T.subtype).hom.comp (CategoryTheory. …
          -/
        · rw [← hom_comp, ofMkLEMk_comp]
          /-
            case h.e'_3
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S T : Submodule R ↑M
            h : LE.le ({ toFun := fun N => CategoryTheory.Subobject.mk (ModuleCat.ofHom N. …
            ⊢ Eq S (LinearMap.range (ModuleCat.ofHom S.subtype).hom)
          -/
          exact (Submodule.range_subtype _).symm
          /-
            🎉 no goals
          -/
          /-
            case h.e'_4
            R : Type u
            inst✝ : Ring R
            M : ModuleCat R
            S T : Submodule R ↑M
            h : LE.le ({ toFun := fun N => CategoryTheory.Subobject.mk (ModuleCat.ofHom N. …
            ⊢ Eq T (LinearMap.range (ModuleCat.ofHom T.subtype).hom)
          -/
        · exact (Submodule.range_subtype _).symm }
          /-
            🎉 no goals
          -/


instance wellPowered_moduleCat : WellPowered.{v} (ModuleCat.{v} R) :=
  ⟨fun M => ⟨⟨_, ⟨(subobjectModule M).toEquiv⟩⟩⟩⟩


/-- Bundle an element `m : M` such that `f m = 0` as a term of `kernelSubobject f`. -/
noncomputable def toKernelSubobject {M N : ModuleCat.{v} R} {f : M ⟶ N} :
    LinearMap.ker f.hom →ₗ[R] kernelSubobject f :=
  (kernelSubobjectIso f ≪≫ ModuleCat.kernelIsoKer f).inv.hom


@[simp]
theorem toKernelSubobject_arrow {M N : ModuleCat R} {f : M ⟶ N} (x : LinearMap.ker f.hom) :
    (kernelSubobject f).arrow (toKernelSubobject x) = x.1 := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10959): the whole proof was just `simp [toKernelSubobject]`.
  suffices ((arrow ((kernelSubobject f))) ∘ (kernelSubobjectIso f ≪≫ kernelIsoKer f).inv) x = x by
    convert this
  /-
    R : Type u
    inst✝ : Ring R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : Subtype fun x => Membership.mem (LinearMap.ker f.hom) x
    ⊢ Eq (Function.comp (⇑(CategoryTheory.Limits.kernelSubobject f).arrow.hom) (⇑( …
  -/
  rw [Iso.trans_inv, ← LinearMap.coe_comp, ← hom_comp, Category.assoc]
  /-
    R : Type u
    inst✝ : Ring R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : Subtype fun x => Membership.mem (LinearMap.ker f.hom) x
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.kernelIsoKer f).inv (Cate …
  -/
  simp only [Category.assoc, kernelSubobject_arrow', kernelIsoKer_inv_kernel_ι]
  /-
    R : Type u
    inst✝ : Ring R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : Subtype fun x => Membership.mem (LinearMap.ker f.hom) x
    ⊢ Eq ((LinearMap.ker f.hom).subtype x) ↑x
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- An extensionality lemma showing that two elements of a cokernel by an image
are equal if they differ by an element of the image.

The application is for homology:
two elements in homology are equal if they differ by a boundary.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/pull/11215): TODO compiler complains that this is marked with `@[ext]`.
-- Should this be changed?
-- @[ext] this is no longer an ext lemma under the current interpretation see eg
-- the conversation beginning at
-- https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/
-- Goal.20state.20not.20updating.2C.20bugs.2C.20etc.2E/near/338456803
theorem cokernel_π_imageSubobject_ext {L M N : ModuleCat.{v} R} (f : L ⟶ M) [HasImage f]
    (g : (imageSubobject f : ModuleCat.{v} R) ⟶ N) [HasCokernel g] {x y : N} (l : L)
    (w : x = y + g (factorThruImageSubobject f l)) : cokernel.π g x = cokernel.π g y := by
  /-
    R : Type u
    inst✝² : Ring R
    L M N : ModuleCat R
    f : Quiver.Hom L M
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj (CategoryTheory.Limits …
    inst✝ : CategoryTheory.Limits.HasCokernel g
    x y : ↑N
    l : ↑L
    w : Eq x (HAdd.hAdd y (g.hom ((CategoryTheory.Limits.factorThruImageSubobject  …
    ⊢ Eq ((CategoryTheory.Limits.cokernel.π g).hom x) ((CategoryTheory.Limits.coke …
  -/
  subst w
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10959): The proof from here used to just be `simp`.
  /-
    R : Type u
    inst✝² : Ring R
    L M N : ModuleCat R
    f : Quiver.Hom L M
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj (CategoryTheory.Limits …
    inst✝ : CategoryTheory.Limits.HasCokernel g
    y : ↑N
    l : ↑L
    ⊢ Eq ((CategoryTheory.Limits.cokernel.π g).hom (HAdd.hAdd y (g.hom ((CategoryT …
  -/
  simp only [map_add, add_right_eq_self]
  -- TODO: add a `@[simp]` lemma along the lines of:
  -- ```
  -- lemma ModuleCat.Hom.cokernel_condition : (cokernel.π g).hom (g.hom x) = 0
  -- ```
  -- ideally generated for all concrete categories (using a metaprogram like `@[elementwise]`?).
  -- See also: https://github.com/leanprover-community/mathlib4/pull/19511#discussion_r1867083077
  /-
    R : Type u
    inst✝² : Ring R
    L M N : ModuleCat R
    f : Quiver.Hom L M
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj (CategoryTheory.Limits …
    inst✝ : CategoryTheory.Limits.HasCokernel g
    y : ↑N
    l : ↑L
    ⊢ Eq ((CategoryTheory.Limits.cokernel.π g).hom (g.hom ((CategoryTheory.Limits. …
  -/
  change ((factorThruImageSubobject f) ≫ g ≫ (cokernel.π g)).hom l = 0
  /-
    R : Type u
    inst✝² : Ring R
    L M N : ModuleCat R
    f : Quiver.Hom L M
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom (CategoryTheory.Subobject.underlying.obj (CategoryTheory.Limits …
    inst✝ : CategoryTheory.Limits.HasCokernel g
    y : ↑N
    l : ↑L
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruIma …
  -/
  simp
  /-
    🎉 no goals
  -/


