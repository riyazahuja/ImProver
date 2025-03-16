/-- The kernel cone induced by the concrete kernel. -/
def kernelCone : KernelFork f :=
  -- Porting note: previously proven by tidy
                                                 /-
                                                   R : Type u
                                                   inst✝ : Ring R
                                                   M N : ModuleCat R
                                                   f : Quiver.Hom M N
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (LinearMap.ker f.hom …
                                                 -/
  KernelFork.ofι (ofHom f.hom.ker.subtype) <| by ext x; cases x; assumption
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- The kernel of a linear map is a kernel in the categorical sense. -/
def kernelIsLimit : IsLimit (kernelCone f) :=
  Fork.IsLimit.mk _
    (fun s => ofHom <|
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation on LinearMap.ker
      LinearMap.codRestrict (LinearMap.ker f.hom) (Fork.ι s).hom fun c =>
        LinearMap.mem_ker.2 <| by
          -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
          /-
            R : Type u
            inst✝ : Ring R
            M N : ModuleCat R
            f : Quiver.Hom M N
            s : CategoryTheory.Limits.Fork f 0
            c : ↑s.1
            ⊢ Eq (f.hom (s.ι.hom c)) 0
          -/
          erw [← @Function.comp_apply _ _ _ f (Fork.ι s) c, ← LinearMap.coe_comp]
          /-
            R : Type u
            inst✝ : Ring R
            M N : ModuleCat R
            f : Quiver.Hom M N
            s : CategoryTheory.Limits.Fork f 0
            c : ↑s.1
            ⊢ Eq ((f.hom.comp s.ι.hom) c) 0
          -/
          rw [← ModuleCat.hom_comp, Fork.condition, HasZeroMorphisms.comp_zero (Fork.ι s) N]
          /-
            R : Type u
            inst✝ : Ring R
            M N : ModuleCat R
            f : Quiver.Hom M N
            s : CategoryTheory.Limits.Fork f 0
            c : ↑s.1
            ⊢ Eq ((ModuleCat.Hom.hom 0) c) 0
          -/
          rfl)
          /-
            🎉 no goals
          -/
    (fun _ => hom_ext <| LinearMap.subtype_comp_codRestrict _ _ _) fun s m h =>
                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : Ring R
                                                                    M N : ModuleCat R
                                                                    f : Quiver.Hom M N
                                                                    s : CategoryTheory.Limits.Fork f 0
                                                                    m : Quiver.Hom s.pt (ModuleCat.kernelCone f).pt
                                                                    h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Mo …
                                                                    x : ↑s.pt
                                                                    ⊢ Eq ↑(m.hom x) ↑(((fun s => ModuleCat.ofHom (LinearMap.codRestrict (LinearMap …
                                                                  -/
      hom_ext <| LinearMap.ext fun x => Subtype.ext_iff_val.2 (by simp [← h]; rfl)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- The cokernel cocone induced by the projection onto the quotient. -/
def cokernelCocone : CokernelCofork f :=
  CokernelCofork.ofπ (ofHom f.hom.range.mkQ) <| hom_ext <| LinearMap.range_mkQ_comp _


/-- The projection onto the quotient is a cokernel in the categorical sense. -/
def cokernelIsColimit : IsColimit (cokernelCocone f) :=
  Cofork.IsColimit.mk _
    (fun s => ofHom <| f.hom.range.liftQ (Cofork.π s).hom <|
      LinearMap.range_le_ker_iff.2 <| ModuleCat.hom_ext_iff.mp <| CokernelCofork.condition s)
    (fun s => hom_ext <| f.hom.range.liftQ_mkQ (Cofork.π s).hom _) fun s m h => by
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation
    haveI : Epi (ofHom (LinearMap.range f.hom).mkQ) :=
      (epi_iff_range_eq_top _).mpr (Submodule.range_mkQ _)
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation
    /-
      R : Type u
      inst✝ : Ring R
      M N : ModuleCat R
      f : Quiver.Hom M N
      s : CategoryTheory.Limits.Cofork f 0
      m : Quiver.Hom (ModuleCat.cokernelCocone f).pt s.pt
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Mo …
      this : CategoryTheory.Epi (ModuleCat.ofHom (LinearMap.range f.hom).mkQ)
      ⊢ Eq m ((fun s => ModuleCat.ofHom ((LinearMap.range f.hom).liftQ s.π.hom ⋯)) s)
    -/
    apply (cancel_epi (ofHom (LinearMap.range f.hom).mkQ)).1
    /-
      R : Type u
      inst✝ : Ring R
      M N : ModuleCat R
      f : Quiver.Hom M N
      s : CategoryTheory.Limits.Cofork f 0
      m : Quiver.Hom (ModuleCat.cokernelCocone f).pt s.pt
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Mo …
      this : CategoryTheory.Epi (ModuleCat.ofHom (LinearMap.range f.hom).mkQ)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (LinearMap.range f.h …
    -/
    exact h
    /-
      🎉 no goals
    -/


/-- The category of R-modules has kernels, given by the inclusion of the kernel submodule. -/
theorem hasKernels_moduleCat : HasKernels (ModuleCat R) :=
  ⟨fun f => HasLimit.mk ⟨_, kernelIsLimit f⟩⟩


/-- The category of R-modules has cokernels, given by the projection onto the quotient. -/
theorem hasCokernels_moduleCat : HasCokernels (ModuleCat R) :=
  ⟨fun f => HasColimit.mk ⟨_, cokernelIsColimit f⟩⟩


/-- The categorical kernel of a morphism in `ModuleCat`
agrees with the usual module-theoretical kernel.
-/
noncomputable def kernelIsoKer {G H : ModuleCat.{v} R} (f : G ⟶ H) :
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation
    kernel f ≅ ModuleCat.of R (LinearMap.ker f.hom) :=
  limit.isoLimitCone ⟨_, kernelIsLimit f⟩

-- We now show this isomorphism commutes with the inclusion of the kernel into the source.

@[simp, elementwise]
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation
theorem kernelIsoKer_inv_kernel_ι : (kernelIsoKer f).inv ≫ kernel.ι f =
    ofHom (LinearMap.ker f.hom).subtype :=
  limit.isoLimitCone_inv_π _ _


@[simp, elementwise]
theorem kernelIsoKer_hom_ker_subtype :
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation
    (kernelIsoKer f).hom ≫ ofHom (LinearMap.ker f.hom).subtype = kernel.ι f :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ (limit.isLimit _) WalkingParallelPair.zero


/-- The categorical cokernel of a morphism in `ModuleCat`
agrees with the usual module-theoretical quotient.
-/
noncomputable def cokernelIsoRangeQuotient {G H : ModuleCat.{v} R} (f : G ⟶ H) :
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11036): broken dot notation
    cokernel f ≅ ModuleCat.of R (H ⧸ LinearMap.range f.hom) :=
  colimit.isoColimitCocone ⟨_, cokernelIsColimit f⟩

-- We now show this isomorphism commutes with the projection of target to the cokernel.

@[simp, elementwise]
theorem cokernel_π_cokernelIsoRangeQuotient_hom :
    cokernel.π f ≫ (cokernelIsoRangeQuotient f).hom = ofHom f.hom.range.mkQ :=
  colimit.isoColimitCocone_ι_hom _ _


@[simp, elementwise]
theorem range_mkQ_cokernelIsoRangeQuotient_inv :
    ofHom f.hom.range.mkQ ≫ (cokernelIsoRangeQuotient f).inv = cokernel.π f :=
  colimit.isoColimitCocone_ι_inv ⟨_, cokernelIsColimit f⟩ WalkingParallelPair.one


theorem cokernel_π_ext {M N : ModuleCat.{u} R} (f : M ⟶ N) {x y : N} (m : M) (w : x = y + f m) :
    cokernel.π f x = cokernel.π f y := by
  /-
    R : Type u
    inst✝ : Ring R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x y : ↑N
    m : ↑M
    w : Eq x (HAdd.hAdd y (f.hom m))
    ⊢ Eq ((CategoryTheory.Limits.cokernel.π f).hom x) ((CategoryTheory.Limits.coke …
  -/
  subst w
  /-
    R : Type u
    inst✝ : Ring R
    M N : ModuleCat R
    f : Quiver.Hom M N
    y : ↑N
    m : ↑M
    ⊢ Eq ((CategoryTheory.Limits.cokernel.π f).hom (HAdd.hAdd y (f.hom m))) ((Cate …
  -/
  simpa only [map_add, add_right_eq_self] using cokernel.condition_apply f m
  /-
    🎉 no goals
  -/


