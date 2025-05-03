/-- The kernel cone induced by the concrete kernel. -/
def kernelCone : KernelFork f :=
  KernelFork.ofι (Z := of f.ker) f.ker.subtype <| ext fun x => x.casesOn fun _ hx => hx


/-- The kernel of a group homomorphism is a kernel in the categorical sense. -/
def kernelIsLimit : IsLimit <| kernelCone f :=
  Fork.IsLimit.mk _
                  /-
                    G H : AddCommGrp
                    f : Quiver.Hom G H
                    s : CategoryTheory.Limits.Fork f 0
                    ⊢ AddMonoidHom ↑s.pt ↑G
                  -/
    (fun s => (by exact Fork.ι s : _ →+ G).codRestrict _ fun c => mem_ker.mpr <|
                  /-
                    🎉 no goals
                  -/
         /-
           G H : AddCommGrp
           f : Quiver.Hom G H
           s : CategoryTheory.Limits.Fork f 0
           c : ↑s.pt
           ⊢ Eq (f (s.ι c)) 0
         -/
      by exact DFunLike.congr_fun s.condition c)
         /-
           🎉 no goals
         -/
                 /-
                   G H : AddCommGrp
                   f : Quiver.Hom G H
                   x✝ : CategoryTheory.Limits.Fork f 0
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => AddMonoidHom.codRestrict s …
                 -/
    (fun _ => by rfl)
                 /-
                   🎉 no goals
                 -/
                                                             /-
                                                               G H : AddCommGrp
                                                               f : Quiver.Hom G H
                                                               x✝¹ : CategoryTheory.Limits.Fork f 0
                                                               x✝ : Quiver.Hom x✝¹.pt (AddCommGrp.kernelCone f).pt
                                                               h : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.Limits.Fork.ι (A …
                                                               x : ↑x✝¹.pt
                                                               ⊢ Eq ↑(x✝ x) ↑(((fun s => AddMonoidHom.codRestrict s.ι (AddMonoidHom.ker f) ⋯) …
                                                             -/
    (fun _ _ h => ext fun x => Subtype.ext_iff_val.mpr <| by exact DFunLike.congr_fun h x)
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The cokernel cocone induced by the projection onto the quotient. -/
def cokernelCocone : CokernelCofork f :=
  CokernelCofork.ofπ (Z := of <| H ⧸ f.range) (mk' f.range) <| ext fun x =>
    (eq_zero_iff _).mpr ⟨x, rfl⟩


/-- The projection onto the quotient is a cokernel in the categorical sense. -/
def cokernelIsColimit : IsColimit <| cokernelCocone f :=
  Cofork.IsColimit.mk _
    (fun s => lift _ _ <| (range_le_ker_iff _ _).mpr <| CokernelCofork.condition s)
    (fun _ => rfl)
    (fun _ _ h => have : Epi (cokernelCocone f).π := (epi_iff_surjective _).mpr <| mk'_surjective _
                                                 /-
                                                   G H : AddCommGrp
                                                   f : Quiver.Hom G H
                                                   x✝¹ : CategoryTheory.Limits.Cofork f 0
                                                   x✝ : Quiver.Hom (AddCommGrp.cokernelCocone f).pt x✝¹.pt
                                                   h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ad …
                                                   this : CategoryTheory.Epi (CategoryTheory.Limits.Cofork.π (AddCommGrp.cokernel …
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (AddC …
                                                 -/
      (cancel_epi (cokernelCocone f).π).mp <| by simpa only [parallelPair_obj_one] using h)
                                                 /-
                                                   🎉 no goals
                                                 -/


