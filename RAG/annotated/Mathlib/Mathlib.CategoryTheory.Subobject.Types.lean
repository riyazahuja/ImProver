theorem subtype_val_mono {α : Type u} (s : Set α) : Mono (↾(Subtype.val : s → α)) :=
  (mono_iff_injective _).mpr Subtype.val_injective


/-- The category of `MonoOver α`, for `α : Type u`, is equivalent to the partial order `Set α`.
-/
@[simps]
noncomputable def Types.monoOverEquivalenceSet (α : Type u) : MonoOver α ≌ Set α where
  functor :=
    { obj := fun f => Set.range f.1.hom
      map := fun {f g} t =>
        homOfLE
          (by
            /-
              α : Type u
              f g : CategoryTheory.MonoOver α
              t : Quiver.Hom f g
              ⊢ LE.le ((fun f => Set.range f.obj.hom) f) ((fun f => Set.range f.obj.hom) g)
            -/
            rintro a ⟨x, rfl⟩
            /-
              case intro
              α : Type u
              f g : CategoryTheory.MonoOver α
              t : Quiver.Hom f g
              x : (CategoryTheory.Functor.id (Type u)).obj f.obj.left
              ⊢ Membership.mem ((fun f => Set.range f.obj.hom) g) (f.obj.hom x)
            -/
            exact ⟨t.1 x, congr_fun t.w x⟩) }
            /-
              🎉 no goals
            -/
  inverse :=
    { obj := fun s => MonoOver.mk' (Subtype.val : s → α)
                            /-
                              α : Type u
                              s t : Set α
                              b : Quiver.Hom s t
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (fun w => ⟨↑w, ⋯⟩) ((fun s => Categor …
                            -/
      map := fun {s t} b => MonoOver.homMk (fun w => ⟨w.1, Set.mem_of_mem_of_subset w.2 b.le⟩) }
                            /-
                              🎉 no goals
                            -/
  unitIso :=
    /-
      α : Type u
      ⊢ ∀ {X Y : CategoryTheory.MonoOver α} (f : Quiver.Hom X Y), Eq (CategoryTheory …
    -/
      /-
        α : Type u
        f : CategoryTheory.MonoOver α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (Equiv.ofInjective f.obj.hom ⋯).toIso …
      -/
    NatIso.ofComponents fun f =>
      /-
        🎉 no goals
      -/
    /-
      🎉 no goals
    -/
      MonoOver.isoMk (Equiv.ofInjective f.1.hom ((mono_iff_injective _).mp f.2)).toIso
               /-
                 α : Type u
                 ⊢ ∀ {X Y : Set α} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp …
               -/
  counitIso := NatIso.ofComponents fun _ => eqToIso Subtype.range_val
               /-
                 🎉 no goals
               -/


instance : WellPowered.{u} (Type u) :=
  wellPowered_of_essentiallySmall_monoOver fun α =>
    EssentiallySmall.mk' (Types.monoOverEquivalenceSet α)


/-- For `α : Type u`, `Subobject α` is order isomorphic to `Set α`.
-/
noncomputable def Types.subobjectEquivSet (α : Type u) : Subobject α ≃o Set α :=
  (Types.monoOverEquivalenceSet α).thinSkeletonOrderIso

