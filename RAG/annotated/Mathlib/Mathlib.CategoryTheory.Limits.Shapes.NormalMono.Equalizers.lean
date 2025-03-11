/-- The pullback of two monomorphisms exists. -/
@[irreducible, nolint defLemma] -- Porting note: changed to irreducible and a def
def pullback_of_mono {X Y Z : C} (a : X ⟶ Z) (b : Y ⟶ Z) [Mono a] [Mono b] :
    HasLimit (cospan a b) :=
  let ⟨P, f, haf, i⟩ := normalMonoOfMono a
  let ⟨Q, g, hbg, i'⟩ := normalMonoOfMono b
  let ⟨a', ha'⟩ :=
    KernelFork.IsLimit.lift' i (kernel.ι (prod.lift f g)) <|
      calc kernel.ι (prod.lift f g) ≫ f
                                                                             /-
                                                                               C : Type u_1
                                                                               inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                                               inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                               inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                                               inst✝³ : CategoryTheory.Limits.HasKernels C
                                                                               inst✝² : CategoryTheory.NormalMonoCategory C
                                                                               X Y Z : C
                                                                               a : Quiver.Hom X Z
                                                                               b : Quiver.Hom Y Z
                                                                               inst✝¹ : CategoryTheory.Mono a
                                                                               inst✝ : CategoryTheory.Mono b
                                                                               P : C
                                                                               f : Quiver.Hom Z P
                                                                               haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                                               i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                                               Q : C
                                                                               g : Quiver.Hom Z Q
                                                                               hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                                               i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (Cate …
                                                                             -/
        _ = kernel.ι (prod.lift f g) ≫ prod.lift f g ≫ Limits.prod.fst := by rw [prod.lift_fst]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                                                         /-
                                                                           C : Type u_1
                                                                           inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                                           inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                           inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                                           inst✝³ : CategoryTheory.Limits.HasKernels C
                                                                           inst✝² : CategoryTheory.NormalMonoCategory C
                                                                           X Y Z : C
                                                                           a : Quiver.Hom X Z
                                                                           b : Quiver.Hom Y Z
                                                                           inst✝¹ : CategoryTheory.Mono a
                                                                           inst✝ : CategoryTheory.Mono b
                                                                           P : C
                                                                           f : Quiver.Hom Z P
                                                                           haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                                           i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                                           Q : C
                                                                           g : Quiver.Hom Z Q
                                                                           hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                                           i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (Cate …
                                                                         -/
        _ = (0 : kernel (prod.lift f g) ⟶ P ⨯ Q) ≫ Limits.prod.fst := by rw [kernel.condition_assoc]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
        _ = 0 := zero_comp

  let ⟨b', hb'⟩ :=
    KernelFork.IsLimit.lift' i' (kernel.ι (prod.lift f g)) <|
      calc kernel.ι (prod.lift f g) ≫ g
                                                                             /-
                                                                               C : Type u_1
                                                                               inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                                               inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                               inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                                               inst✝³ : CategoryTheory.Limits.HasKernels C
                                                                               inst✝² : CategoryTheory.NormalMonoCategory C
                                                                               X Y Z : C
                                                                               a : Quiver.Hom X Z
                                                                               b : Quiver.Hom Y Z
                                                                               inst✝¹ : CategoryTheory.Mono a
                                                                               inst✝ : CategoryTheory.Mono b
                                                                               P : C
                                                                               f : Quiver.Hom Z P
                                                                               haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                                               i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                                               Q : C
                                                                               g : Quiver.Hom Z Q
                                                                               hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                                               i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                                               a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                                               ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (Cate …
                                                                             -/
        _ = kernel.ι (prod.lift f g) ≫ prod.lift f g ≫ Limits.prod.snd := by rw [prod.lift_snd]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                                                                         /-
                                                                           C : Type u_1
                                                                           inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                                           inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                           inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                                           inst✝³ : CategoryTheory.Limits.HasKernels C
                                                                           inst✝² : CategoryTheory.NormalMonoCategory C
                                                                           X Y Z : C
                                                                           a : Quiver.Hom X Z
                                                                           b : Quiver.Hom Y Z
                                                                           inst✝¹ : CategoryTheory.Mono a
                                                                           inst✝ : CategoryTheory.Mono b
                                                                           P : C
                                                                           f : Quiver.Hom Z P
                                                                           haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                                           i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                                           Q : C
                                                                           g : Quiver.Hom Z Q
                                                                           hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                                           i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                                           a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                                           ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (Cate …
                                                                         -/
        _ = (0 : kernel (prod.lift f g) ⟶ P ⨯ Q) ≫ Limits.prod.snd := by rw [kernel.condition_assoc]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
        _ = 0 := zero_comp

  HasLimit.mk
    { cone :=
        PullbackCone.mk a' b' <| by
          simp? at ha' hb' says
            simp only [parallelPair_obj_zero, Fork.ofι_pt, Fork.ι_ofι] at ha' hb'
          /-
            C : Type u_1
            inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.NormalMonoCategory C
            X Y Z : C
            a : Quiver.Hom X Z
            b : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.Mono a
            inst✝ : CategoryTheory.Mono b
            P : C
            f : Quiver.Hom Z P
            haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
            i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
            Q : C
            g : Quiver.Hom Z Q
            hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
            i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
            a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
            ha' : Eq (CategoryTheory.CategoryStruct.comp a' a) (CategoryTheory.Limits.kern …
            b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
            hb' : Eq (CategoryTheory.CategoryStruct.comp b' b) (CategoryTheory.Limits.kern …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp a' a) (CategoryTheory.CategoryStruct. …
          -/
          rw [ha', hb']
          /-
            🎉 no goals
          -/
      isLimit :=
        PullbackCone.IsLimit.mk _
          (fun s =>
            kernel.lift (prod.lift f g) (PullbackCone.snd s ≫ b) <|
              Limits.prod.hom_ext
                (calc
                  ((PullbackCone.snd s ≫ b) ≫ prod.lift f g) ≫ Limits.prod.fst =
                                                       /-
                                                         C : Type u_1
                                                         inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                         inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                         inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                         inst✝³ : CategoryTheory.Limits.HasKernels C
                                                         inst✝² : CategoryTheory.NormalMonoCategory C
                                                         X Y Z : C
                                                         a : Quiver.Hom X Z
                                                         b : Quiver.Hom Y Z
                                                         inst✝¹ : CategoryTheory.Mono a
                                                         inst✝ : CategoryTheory.Mono b
                                                         P : C
                                                         f : Quiver.Hom Z P
                                                         haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                         i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                         Q : C
                                                         g : Quiver.Hom Z Q
                                                         hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                         i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                         a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                         ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                         b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                         hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                         s : CategoryTheory.Limits.PullbackCone a b
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                       -/
                      PullbackCone.snd s ≫ b ≫ f := by simp only [prod.lift_fst, Category.assoc]
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                       /-
                                                         C : Type u_1
                                                         inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                         inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                         inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                         inst✝³ : CategoryTheory.Limits.HasKernels C
                                                         inst✝² : CategoryTheory.NormalMonoCategory C
                                                         X Y Z : C
                                                         a : Quiver.Hom X Z
                                                         b : Quiver.Hom Y Z
                                                         inst✝¹ : CategoryTheory.Mono a
                                                         inst✝ : CategoryTheory.Mono b
                                                         P : C
                                                         f : Quiver.Hom Z P
                                                         haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                         i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                         Q : C
                                                         g : Quiver.Hom Z Q
                                                         hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                         i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                         a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                         ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                         b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                         hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                         s : CategoryTheory.Limits.PullbackCone a b
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp s.snd (CategoryTheory.CategoryStruct. …
                                                       -/
                  _ = PullbackCone.fst s ≫ a ≫ f := by rw [PullbackCone.condition_assoc]
                                                       /-
                                                         🎉 no goals
                                                       -/
                                                   /-
                                                     C : Type u_1
                                                     inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                     inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                     inst✝³ : CategoryTheory.Limits.HasKernels C
                                                     inst✝² : CategoryTheory.NormalMonoCategory C
                                                     X Y Z : C
                                                     a : Quiver.Hom X Z
                                                     b : Quiver.Hom Y Z
                                                     inst✝¹ : CategoryTheory.Mono a
                                                     inst✝ : CategoryTheory.Mono b
                                                     P : C
                                                     f : Quiver.Hom Z P
                                                     haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                     i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                     Q : C
                                                     g : Quiver.Hom Z Q
                                                     hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                     i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                     a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                     ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                     b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                     hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                     s : CategoryTheory.Limits.PullbackCone a b
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp s.fst (CategoryTheory.CategoryStruct. …
                                                   -/
                  _ = PullbackCone.fst s ≫ 0 := by rw [haf]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                /-
                                                  C : Type u_1
                                                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                  inst✝³ : CategoryTheory.Limits.HasKernels C
                                                  inst✝² : CategoryTheory.NormalMonoCategory C
                                                  X Y Z : C
                                                  a : Quiver.Hom X Z
                                                  b : Quiver.Hom Y Z
                                                  inst✝¹ : CategoryTheory.Mono a
                                                  inst✝ : CategoryTheory.Mono b
                                                  P : C
                                                  f : Quiver.Hom Z P
                                                  haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                  i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                  Q : C
                                                  g : Quiver.Hom Z Q
                                                  hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                  i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                  a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                  ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                  b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                  hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                  s : CategoryTheory.Limits.PullbackCone a b
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp s.fst 0) (CategoryTheory.CategoryStru …
                                                -/
                  _ = 0 ≫ Limits.prod.fst := by rw [comp_zero, zero_comp]
                                                /-
                                                  🎉 no goals
                                                -/
                  )
                (calc
                  ((PullbackCone.snd s ≫ b) ≫ prod.lift f g) ≫ Limits.prod.snd =
                      PullbackCone.snd s ≫ b ≫ g := by
                    /-
                      C : Type u_1
                      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                      inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                      inst✝³ : CategoryTheory.Limits.HasKernels C
                      inst✝² : CategoryTheory.NormalMonoCategory C
                      X Y Z : C
                      a : Quiver.Hom X Z
                      b : Quiver.Hom Y Z
                      inst✝¹ : CategoryTheory.Mono a
                      inst✝ : CategoryTheory.Mono b
                      P : C
                      f : Quiver.Hom Z P
                      haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                      i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                      Q : C
                      g : Quiver.Hom Z Q
                      hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                      i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                      a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                      ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                      b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                      hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                      s : CategoryTheory.Limits.PullbackCone a b
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                    -/
                    simp only [prod.lift_snd, Category.assoc]
                    /-
                      🎉 no goals
                    -/
                                                   /-
                                                     C : Type u_1
                                                     inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                     inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                     inst✝³ : CategoryTheory.Limits.HasKernels C
                                                     inst✝² : CategoryTheory.NormalMonoCategory C
                                                     X Y Z : C
                                                     a : Quiver.Hom X Z
                                                     b : Quiver.Hom Y Z
                                                     inst✝¹ : CategoryTheory.Mono a
                                                     inst✝ : CategoryTheory.Mono b
                                                     P : C
                                                     f : Quiver.Hom Z P
                                                     haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                     i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                     Q : C
                                                     g : Quiver.Hom Z Q
                                                     hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                     i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                     a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                     ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                     b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                     hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                     s : CategoryTheory.Limits.PullbackCone a b
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp s.snd (CategoryTheory.CategoryStruct. …
                                                   -/
                  _ = PullbackCone.snd s ≫ 0 := by rw [hbg]
                                                   /-
                                                     🎉 no goals
                                                   -/
                                                /-
                                                  C : Type u_1
                                                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                  inst✝³ : CategoryTheory.Limits.HasKernels C
                                                  inst✝² : CategoryTheory.NormalMonoCategory C
                                                  X Y Z : C
                                                  a : Quiver.Hom X Z
                                                  b : Quiver.Hom Y Z
                                                  inst✝¹ : CategoryTheory.Mono a
                                                  inst✝ : CategoryTheory.Mono b
                                                  P : C
                                                  f : Quiver.Hom Z P
                                                  haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                  i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                  Q : C
                                                  g : Quiver.Hom Z Q
                                                  hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                  i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                  a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                  ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                  b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                  hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                  s : CategoryTheory.Limits.PullbackCone a b
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp s.snd 0) (CategoryTheory.CategoryStru …
                                                -/
                  _ = 0 ≫ Limits.prod.snd := by rw [comp_zero, zero_comp]
                                                /-
                                                  🎉 no goals
                                                -/
                  ))
          (fun s =>
            (cancel_mono a).1 <| by
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝³ : CategoryTheory.Limits.HasKernels C
                inst✝² : CategoryTheory.NormalMonoCategory C
                X Y Z : C
                a : Quiver.Hom X Z
                b : Quiver.Hom Y Z
                inst✝¹ : CategoryTheory.Mono a
                inst✝ : CategoryTheory.Mono b
                P : C
                f : Quiver.Hom Z P
                haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                Q : C
                g : Quiver.Hom Z Q
                hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                s : CategoryTheory.Limits.PullbackCone a b
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              rw [KernelFork.ι_ofι] at ha'
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝³ : CategoryTheory.Limits.HasKernels C
                inst✝² : CategoryTheory.NormalMonoCategory C
                X Y Z : C
                a : Quiver.Hom X Z
                b : Quiver.Hom Y Z
                inst✝¹ : CategoryTheory.Mono a
                inst✝ : CategoryTheory.Mono b
                P : C
                f : Quiver.Hom Z P
                haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                Q : C
                g : Quiver.Hom Z Q
                hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                ha' : Eq (CategoryTheory.CategoryStruct.comp a' a) (CategoryTheory.Limits.kern …
                b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                s : CategoryTheory.Limits.PullbackCone a b
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              simp [ha', PullbackCone.condition s])
              /-
                🎉 no goals
              -/
          (fun s =>
            (cancel_mono b).1 <| by
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝³ : CategoryTheory.Limits.HasKernels C
                inst✝² : CategoryTheory.NormalMonoCategory C
                X Y Z : C
                a : Quiver.Hom X Z
                b : Quiver.Hom Y Z
                inst✝¹ : CategoryTheory.Mono a
                inst✝ : CategoryTheory.Mono b
                P : C
                f : Quiver.Hom Z P
                haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                Q : C
                g : Quiver.Hom Z Q
                hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                s : CategoryTheory.Limits.PullbackCone a b
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              rw [KernelFork.ι_ofι] at hb'
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                inst✝³ : CategoryTheory.Limits.HasKernels C
                inst✝² : CategoryTheory.NormalMonoCategory C
                X Y Z : C
                a : Quiver.Hom X Z
                b : Quiver.Hom Y Z
                inst✝¹ : CategoryTheory.Mono a
                inst✝ : CategoryTheory.Mono b
                P : C
                f : Quiver.Hom Z P
                haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                Q : C
                g : Quiver.Hom Z Q
                hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                hb' : Eq (CategoryTheory.CategoryStruct.comp b' b) (CategoryTheory.Limits.kern …
                s : CategoryTheory.Limits.PullbackCone a b
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
              -/
              simp [hb'])
              /-
                🎉 no goals
              -/
          fun s m h₁ _ =>
          (cancel_mono (kernel.ι (prod.lift f g))).1 <|
            calc
              m ≫ kernel.ι (prod.lift f g) = m ≫ a' ≫ a := by
                /-
                  C : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                  inst✝³ : CategoryTheory.Limits.HasKernels C
                  inst✝² : CategoryTheory.NormalMonoCategory C
                  X Y Z : C
                  a : Quiver.Hom X Z
                  b : Quiver.Hom Y Z
                  inst✝¹ : CategoryTheory.Mono a
                  inst✝ : CategoryTheory.Mono b
                  P : C
                  f : Quiver.Hom Z P
                  haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                  i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                  Q : C
                  g : Quiver.Hom Z Q
                  hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                  i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                  a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                  ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                  b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                  hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                  s : CategoryTheory.Limits.PullbackCone a b
                  m : Quiver.Hom s.pt (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod. …
                  h₁ : Eq (CategoryTheory.CategoryStruct.comp m a') s.fst
                  x✝ : Eq (CategoryTheory.CategoryStruct.comp m b') s.snd
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.kernel.ι (Ca …
                -/
                congr
                /-
                  case e_a
                  C : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                  inst✝³ : CategoryTheory.Limits.HasKernels C
                  inst✝² : CategoryTheory.NormalMonoCategory C
                  X Y Z : C
                  a : Quiver.Hom X Z
                  b : Quiver.Hom Y Z
                  inst✝¹ : CategoryTheory.Mono a
                  inst✝ : CategoryTheory.Mono b
                  P : C
                  f : Quiver.Hom Z P
                  haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                  i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                  Q : C
                  g : Quiver.Hom Z Q
                  hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                  i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                  a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                  ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                  b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                  hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                  s : CategoryTheory.Limits.PullbackCone a b
                  m : Quiver.Hom s.pt (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod. …
                  h₁ : Eq (CategoryTheory.CategoryStruct.comp m a') s.fst
                  x✝ : Eq (CategoryTheory.CategoryStruct.comp m b') s.snd
                  ⊢ Eq (CategoryTheory.Limits.kernel.ι (CategoryTheory.Limits.prod.lift f g)) (C …
                -/
                exact ha'.symm
                /-
                  🎉 no goals
                -/
                                               /-
                                                 C : Type u_1
                                                 inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                 inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                 inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                 inst✝³ : CategoryTheory.Limits.HasKernels C
                                                 inst✝² : CategoryTheory.NormalMonoCategory C
                                                 X Y Z : C
                                                 a : Quiver.Hom X Z
                                                 b : Quiver.Hom Y Z
                                                 inst✝¹ : CategoryTheory.Mono a
                                                 inst✝ : CategoryTheory.Mono b
                                                 P : C
                                                 f : Quiver.Hom Z P
                                                 haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                 i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                 Q : C
                                                 g : Quiver.Hom Z Q
                                                 hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                 i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                 a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                 ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                 b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                 hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                 s : CategoryTheory.Limits.PullbackCone a b
                                                 m : Quiver.Hom s.pt (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod. …
                                                 h₁ : Eq (CategoryTheory.CategoryStruct.comp m a') s.fst
                                                 x✝ : Eq (CategoryTheory.CategoryStruct.comp m b') s.snd
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.comp …
                                               -/
              _ = PullbackCone.fst s ≫ a := by rw [← Category.assoc, h₁]
                                               /-
                                                 🎉 no goals
                                               -/
              _ = PullbackCone.snd s ≫ b := PullbackCone.condition s
              _ =
                  kernel.lift (prod.lift f g) (PullbackCone.snd s ≫ b) _ ≫
                                                   /-
                                                     C : Type u_1
                                                     inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                     inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     inst✝⁴ : CategoryTheory.Limits.HasFiniteProducts C
                                                     inst✝³ : CategoryTheory.Limits.HasKernels C
                                                     inst✝² : CategoryTheory.NormalMonoCategory C
                                                     X Y Z : C
                                                     a : Quiver.Hom X Z
                                                     b : Quiver.Hom Y Z
                                                     inst✝¹ : CategoryTheory.Mono a
                                                     inst✝ : CategoryTheory.Mono b
                                                     P : C
                                                     f : Quiver.Hom Z P
                                                     haf : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                                                     i : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι a haf)
                                                     Q : C
                                                     g : Quiver.Hom Z Q
                                                     hbg : Eq (CategoryTheory.CategoryStruct.comp b g) 0
                                                     i' : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι b hbg)
                                                     a' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                     ha' : Eq (CategoryTheory.CategoryStruct.comp a' (CategoryTheory.Limits.Fork.ι  …
                                                     b' : Quiver.Hom (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod.lift …
                                                     hb' : Eq (CategoryTheory.CategoryStruct.comp b' (CategoryTheory.Limits.Fork.ι  …
                                                     s : CategoryTheory.Limits.PullbackCone a b
                                                     m : Quiver.Hom s.pt (CategoryTheory.Limits.kernel (CategoryTheory.Limits.prod. …
                                                     h₁ : Eq (CategoryTheory.CategoryStruct.comp m a') s.fst
                                                     x✝ : Eq (CategoryTheory.CategoryStruct.comp m b') s.snd
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp s.snd b) (CategoryTheory.CategoryStru …
                                                   -/
                    kernel.ι (prod.lift f g) := by rw [kernel.lift_ι]
                                                   /-
                                                     🎉 no goals
                                                   -/
               }


/-- The pullback of `(𝟙 X, f)` and `(𝟙 X, g)` -/
private abbrev P {X Y : C} (f g : X ⟶ Y) [Mono (prod.lift (𝟙 X) f)] [Mono (prod.lift (𝟙 X) g)] :
    C :=
  pullback (prod.lift (𝟙 X) f) (prod.lift (𝟙 X) g)


/-- The equalizer of `f` and `g` exists. -/
 -- Porting note: changed to irreducible def since irreducible_def was breaking things
@[irreducible, nolint defLemma]
def hasLimit_parallelPair {X Y : C} (f g : X ⟶ Y) : HasLimit (parallelPair f g) :=
  have huv : (pullback.fst _ _ : P f g ⟶ X) = pullback.snd _ _ :=
    calc
      (pullback.fst _ _ : P f g ⟶ X) = pullback.fst _ _ ≫ 𝟙 _ := Eq.symm <| Category.comp_id _
                                                                       /-
                                                                         C : Type u_1
                                                                         inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                         inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                                                         inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                         inst✝ : CategoryTheory.NormalMonoCategory C
                                                                         X Y : C
                                                                         f g : Quiver.Hom X Y
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
                                                                       -/
      _ = pullback.fst _ _ ≫ prod.lift (𝟙 X) f ≫ Limits.prod.fst := by rw [prod.lift_fst]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         C : Type u_1
                                                                         inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                         inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                                                         inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                         inst✝ : CategoryTheory.NormalMonoCategory C
                                                                         X Y : C
                                                                         f g : Quiver.Hom X Y
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
                                                                       -/
      _ = pullback.snd _ _ ≫ prod.lift (𝟙 X) g ≫ Limits.prod.fst := by rw [pullback.condition_assoc]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                 /-
                                   C : Type u_1
                                   inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                   inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                   inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                   inst✝¹ : CategoryTheory.Limits.HasKernels C
                                   inst✝ : CategoryTheory.NormalMonoCategory C
                                   X Y : C
                                   f g : Quiver.Hom X Y
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
                                 -/
      _ = pullback.snd _ _ := by rw [prod.lift_fst, Category.comp_id]
                                 /-
                                   🎉 no goals
                                 -/

  have hvu : (pullback.fst _ _ : P f g ⟶ X) ≫ f = pullback.snd _ _ ≫ g :=
    calc
      (pullback.fst _ _ : P f g ⟶ X) ≫ f =
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                       inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                                                       inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                       inst✝ : CategoryTheory.NormalMonoCategory C
                                                                       X Y : C
                                                                       f g : Quiver.Hom X Y
                                                                       huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
                                                                     -/
        pullback.fst _ _ ≫ prod.lift (𝟙 X) f ≫ Limits.prod.snd := by rw [prod.lift_snd]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                                                       /-
                                                                         C : Type u_1
                                                                         inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                         inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                                                         inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                         inst✝ : CategoryTheory.NormalMonoCategory C
                                                                         X Y : C
                                                                         f g : Quiver.Hom X Y
                                                                         huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
                                                                       -/
      _ = pullback.snd _ _ ≫ prod.lift (𝟙 X) g ≫ Limits.prod.snd := by rw [pullback.condition_assoc]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                     /-
                                       C : Type u_1
                                       inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                       inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                       inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                       inst✝¹ : CategoryTheory.Limits.HasKernels C
                                       inst✝ : CategoryTheory.NormalMonoCategory C
                                       X Y : C
                                       f g : Quiver.Hom X Y
                                       huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
                                     -/
      _ = pullback.snd _ _ ≫ g := by rw [prod.lift_snd]
                                     /-
                                       🎉 no goals
                                     -/

                                                                             /-
                                                                               C : Type u_1
                                                                               inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                               inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                               inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                                                               inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                               inst✝ : CategoryTheory.NormalMonoCategory C
                                                                               X Y : C
                                                                               f g : Quiver.Hom X Y
                                                                               huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                                                                               hvu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
                                                                             -/
  have huu : (pullback.fst _ _ : P f g ⟶ X) ≫ f = pullback.fst _ _ ≫ g := by rw [hvu, ← huv]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  HasLimit.mk
    { cone := Fork.ofι (pullback.fst _ _) huu
      isLimit :=
        Fork.IsLimit.mk _
          (fun s =>
            pullback.lift (Fork.ι s) (Fork.ι s) <|
                                      /-
                                        C : Type u_1
                                        inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                        inst✝¹ : CategoryTheory.Limits.HasKernels C
                                        inst✝ : CategoryTheory.NormalMonoCategory C
                                        X Y : C
                                        f g : Quiver.Hom X Y
                                        huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                                        hvu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                                        huu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                                        s : CategoryTheory.Limits.Fork f g
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
                                      -/
              Limits.prod.hom_ext (by simp only [prod.lift_fst, Category.assoc])
                                      /-
                                        🎉 no goals
                                      -/
                    /-
                      C : Type u_1
                      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                      inst✝¹ : CategoryTheory.Limits.HasKernels C
                      inst✝ : CategoryTheory.NormalMonoCategory C
                      X Y : C
                      f g : Quiver.Hom X Y
                      huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                      hvu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                      huu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                      s : CategoryTheory.Limits.Fork f g
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
                    -/
                (by simp only [prod.comp_lift, Fork.condition s]))
                    /-
                      🎉 no goals
                    -/
                       /-
                         C : Type u_1
                         inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                         inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                         inst✝¹ : CategoryTheory.Limits.HasKernels C
                         inst✝ : CategoryTheory.NormalMonoCategory C
                         X Y : C
                         f g : Quiver.Hom X Y
                         huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                         hvu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                         huu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                         s : CategoryTheory.Limits.Fork f g
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
                       -/
          (fun s => by simp) fun s m h =>
                       /-
                         🎉 no goals
                       -/
                               /-
                                 C : Type u_1
                                 inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                 inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                 inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                 inst✝¹ : CategoryTheory.Limits.HasKernels C
                                 inst✝ : CategoryTheory.NormalMonoCategory C
                                 X Y : C
                                 f g : Quiver.Hom X Y
                                 huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                                 hvu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                                 huu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                                 s : CategoryTheory.Limits.Fork f g
                                 m : Quiver.Hom s.pt (CategoryTheory.Limits.Fork.ofι (CategoryTheory.Limits.pul …
                                 h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι ( …
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.pullback.fst …
                               -/
          pullback.hom_ext (by simpa only [pullback.lift_fst] using h)
                               /-
                                 🎉 no goals
                               -/
                /-
                  C : Type u_1
                  inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                  inst✝¹ : CategoryTheory.Limits.HasKernels C
                  inst✝ : CategoryTheory.NormalMonoCategory C
                  X Y : C
                  f g : Quiver.Hom X Y
                  huv : Eq (CategoryTheory.Limits.pullback.fst (CategoryTheory.Limits.prod.lift  …
                  hvu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                  huu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.f …
                  s : CategoryTheory.Limits.Fork f g
                  m : Quiver.Hom s.pt (CategoryTheory.Limits.Fork.ofι (CategoryTheory.Limits.pul …
                  h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ofι ( …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.pullback.snd …
                -/
            (by simpa only [huv.symm, pullback.lift_fst] using h) }
                /-
                  🎉 no goals
                -/


/-- A `NormalMonoCategory` category with finite products and kernels has all equalizers. -/
instance (priority := 100) hasEqualizers : HasEqualizers C :=
  hasEqualizers_of_hasLimit_parallelPair _


/-- If a zero morphism is a cokernel of `f`, then `f` is an epimorphism. -/
theorem epi_of_zero_cokernel {X Y : C} (f : X ⟶ Y) (Z : C)
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝⁴ : CategoryTheory.Category.{?u.134406, u_1} C
                                                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        inst✝² : CategoryTheory.Limits.HasFiniteProducts C
                                                                        inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                        inst✝ : CategoryTheory.NormalMonoCategory C
                                                                        X Y : C
                                                                        f : Quiver.Hom X Y
                                                                        Z : C
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f 0) 0
                                                                      -/
    (l : IsColimit (CokernelCofork.ofπ (0 : Y ⟶ Z) (show f ≫ 0 = 0 by simp))) : Epi f :=
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  ⟨fun u v huv => by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      ⊢ Eq u v
    -/
    obtain ⟨W, w, hw, hl⟩ := normalMonoOfMono (equalizer.ι u v)
    /-
      case mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      ⊢ Eq u v
    -/
    obtain ⟨m, hm⟩ := equalizer.lift' f huv
    /-
      case mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      m : Quiver.Hom X (CategoryTheory.Limits.equalizer u v)
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer …
      ⊢ Eq u v
    -/
    have hwf : f ≫ w = 0 := by rw [← hm, Category.assoc, hw, comp_zero]
    /-
      case mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      m : Quiver.Hom X (CategoryTheory.Limits.equalizer u v)
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer …
      hwf : Eq (CategoryTheory.CategoryStruct.comp f w) 0
      ⊢ Eq u v
    -/
    obtain ⟨n, hn⟩ := CokernelCofork.IsColimit.desc' l _ hwf
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      m : Quiver.Hom X (CategoryTheory.Limits.equalizer u v)
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer …
      hwf : Eq (CategoryTheory.CategoryStruct.comp f w) 0
      n : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ 0 ⋯).pt W
      hn : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
      ⊢ Eq u v
    -/
    rw [Cofork.π_ofπ, zero_comp] at hn
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      m : Quiver.Hom X (CategoryTheory.Limits.equalizer u v)
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer …
      hwf : Eq (CategoryTheory.CategoryStruct.comp f w) 0
      n : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ 0 ⋯).pt W
      hn : Eq 0 w
      ⊢ Eq u v
    -/
    have : IsIso (equalizer.ι u v) := by apply isIso_limit_cone_parallelPair_of_eq hn.symm hl
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      m : Quiver.Hom X (CategoryTheory.Limits.equalizer u v)
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer …
      hwf : Eq (CategoryTheory.CategoryStruct.comp f w) 0
      n : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ 0 ⋯).pt W
      hn : Eq 0 w
      this : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι u v)
      ⊢ Eq u v
    -/
    apply (cancel_epi (equalizer.ι u v)).1
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteProducts C
      inst✝¹ : CategoryTheory.Limits.HasKernels C
      inst✝ : CategoryTheory.NormalMonoCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
      Z✝ : C
      u v : Quiver.Hom Y Z✝
      huv : Eq (CategoryTheory.CategoryStruct.comp f u) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom Y W
      hw : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι …
      hl : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Cate …
      m : Quiver.Hom X (CategoryTheory.Limits.equalizer u v)
      hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.equalizer …
      hwf : Eq (CategoryTheory.CategoryStruct.comp f w) 0
      n : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ 0 ⋯).pt W
      hn : Eq 0 w
      this : CategoryTheory.IsIso (CategoryTheory.Limits.equalizer.ι u v)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.ι u  …
    -/
    exact equalizer.condition _ _⟩
    /-
      🎉 no goals
    -/


/-- If `f ≫ g = 0` implies `g = 0` for all `g`, then `g` is a monomorphism. -/
theorem epi_of_zero_cancel {X Y : C} (f : X ⟶ Y)
    (hf : ∀ (Z : C) (g : Y ⟶ Z) (_ : f ≫ g = 0), g = 0) : Epi f :=
  epi_of_zero_cokernel f 0 <| zeroCokernelOfZeroCancel f hf


/-- The pushout of two epimorphisms exists. -/
@[irreducible, nolint defLemma] -- Porting note: made a def and re-added irreducible
def pushout_of_epi {X Y Z : C} (a : X ⟶ Y) (b : X ⟶ Z) [Epi a] [Epi b] :
    HasColimit (span a b) :=
  let ⟨P, f, hfa, i⟩ := normalEpiOfEpi a
  let ⟨Q, g, hgb, i'⟩ := normalEpiOfEpi b
  let ⟨a', ha'⟩ :=
    CokernelCofork.IsColimit.desc' i (cokernel.π (coprod.desc f g)) <|
      calc
        f ≫ cokernel.π (coprod.desc f g) =
            coprod.inl ≫ coprod.desc f g ≫ cokernel.π (coprod.desc f g) := by
          /-
            C : Type u_1
            inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝³ : CategoryTheory.Limits.HasCokernels C
            inst✝² : CategoryTheory.NormalEpiCategory C
            X Y Z : C
            a : Quiver.Hom X Y
            b : Quiver.Hom X Z
            inst✝¹ : CategoryTheory.Epi a
            inst✝ : CategoryTheory.Epi b
            P : C
            f : Quiver.Hom P X
            hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
            i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
            Q : C
            g : Quiver.Hom Q X
            hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
            i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.cokernel.π ( …
          -/
          rw [coprod.inl_desc_assoc]
          /-
            🎉 no goals
          -/
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                                          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                          inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                                          inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                                          inst✝² : CategoryTheory.NormalEpiCategory C
                                                                          X Y Z : C
                                                                          a : Quiver.Hom X Y
                                                                          b : Quiver.Hom X Z
                                                                          inst✝¹ : CategoryTheory.Epi a
                                                                          inst✝ : CategoryTheory.Epi b
                                                                          P : C
                                                                          f : Quiver.Hom P X
                                                                          hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                                          i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                                          Q : C
                                                                          g : Quiver.Hom Q X
                                                                          hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                                          i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
                                                                        -/
        _ = coprod.inl ≫ (0 : P ⨿ Q ⟶ cokernel (coprod.desc f g)) := by rw [cokernel.condition]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
        _ = 0 := HasZeroMorphisms.comp_zero _ _

  let ⟨b', hb'⟩ :=
    CokernelCofork.IsColimit.desc' i' (cokernel.π (coprod.desc f g)) <|
      calc
        g ≫ cokernel.π (coprod.desc f g) =
            coprod.inr ≫ coprod.desc f g ≫ cokernel.π (coprod.desc f g) := by
          /-
            C : Type u_1
            inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝³ : CategoryTheory.Limits.HasCokernels C
            inst✝² : CategoryTheory.NormalEpiCategory C
            X Y Z : C
            a : Quiver.Hom X Y
            b : Quiver.Hom X Z
            inst✝¹ : CategoryTheory.Epi a
            inst✝ : CategoryTheory.Epi b
            P : C
            f : Quiver.Hom P X
            hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
            i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
            Q : C
            g : Quiver.Hom Q X
            hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
            i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
            a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
            ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.cokernel.π ( …
          -/
          rw [coprod.inr_desc_assoc]
          /-
            🎉 no goals
          -/
                                                                        /-
                                                                          C : Type u_1
                                                                          inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                                          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                          inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                                          inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                                          inst✝² : CategoryTheory.NormalEpiCategory C
                                                                          X Y Z : C
                                                                          a : Quiver.Hom X Y
                                                                          b : Quiver.Hom X Z
                                                                          inst✝¹ : CategoryTheory.Epi a
                                                                          inst✝ : CategoryTheory.Epi b
                                                                          P : C
                                                                          f : Quiver.Hom P X
                                                                          hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                                          i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                                          Q : C
                                                                          g : Quiver.Hom Q X
                                                                          hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                                          i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                                          a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                                          ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
                                                                        -/
        _ = coprod.inr ≫ (0 : P ⨿ Q ⟶ cokernel (coprod.desc f g)) := by rw [cokernel.condition]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
        _ = 0 := HasZeroMorphisms.comp_zero _ _

  HasColimit.mk
    { cocone :=
        PushoutCocone.mk a' b' <| by
          /-
            C : Type u_1
            inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝³ : CategoryTheory.Limits.HasCokernels C
            inst✝² : CategoryTheory.NormalEpiCategory C
            X Y Z : C
            a : Quiver.Hom X Y
            b : Quiver.Hom X Z
            inst✝¹ : CategoryTheory.Epi a
            inst✝ : CategoryTheory.Epi b
            P : C
            f : Quiver.Hom P X
            hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
            i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
            Q : C
            g : Quiver.Hom Q X
            hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
            i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
            a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
            ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
            b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
            hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp a a') (CategoryTheory.CategoryStruct. …
          -/
          simp only [Cofork.π_ofπ] at ha' hb'
          /-
            C : Type u_1
            inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
            inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
            inst✝³ : CategoryTheory.Limits.HasCokernels C
            inst✝² : CategoryTheory.NormalEpiCategory C
            X Y Z : C
            a : Quiver.Hom X Y
            b : Quiver.Hom X Z
            inst✝¹ : CategoryTheory.Epi a
            inst✝ : CategoryTheory.Epi b
            P : C
            f : Quiver.Hom P X
            hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
            i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
            Q : C
            g : Quiver.Hom Q X
            hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
            i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
            a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
            ha' : Eq (CategoryTheory.CategoryStruct.comp a a') (CategoryTheory.Limits.coke …
            b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
            hb' : Eq (CategoryTheory.CategoryStruct.comp b b') (CategoryTheory.Limits.coke …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp a a') (CategoryTheory.CategoryStruct. …
          -/
          rw [ha', hb']
          /-
            🎉 no goals
          -/
      isColimit :=
        PushoutCocone.IsColimit.mk _
          (fun s =>
            cokernel.desc (coprod.desc f g) (b ≫ PushoutCocone.inr s) <|
              coprod.hom_ext
                (calc
                  coprod.inl ≫ coprod.desc f g ≫ b ≫ PushoutCocone.inr s =
                                                        /-
                                                          C : Type u_1
                                                          inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                          inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                          inst✝² : CategoryTheory.NormalEpiCategory C
                                                          X Y Z : C
                                                          a : Quiver.Hom X Y
                                                          b : Quiver.Hom X Z
                                                          inst✝¹ : CategoryTheory.Epi a
                                                          inst✝ : CategoryTheory.Epi b
                                                          P : C
                                                          f : Quiver.Hom P X
                                                          hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                          i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                          Q : C
                                                          g : Quiver.Hom Q X
                                                          hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                          i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                          a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                          ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                          b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                                          hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                          s : CategoryTheory.Limits.PushoutCocone a b
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
                                                        -/
                      f ≫ b ≫ PushoutCocone.inr s := by rw [coprod.inl_desc_assoc]
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          C : Type u_1
                                                          inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                          inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                          inst✝² : CategoryTheory.NormalEpiCategory C
                                                          X Y Z : C
                                                          a : Quiver.Hom X Y
                                                          b : Quiver.Hom X Z
                                                          inst✝¹ : CategoryTheory.Epi a
                                                          inst✝ : CategoryTheory.Epi b
                                                          P : C
                                                          f : Quiver.Hom P X
                                                          hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                          i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                          Q : C
                                                          g : Quiver.Hom Q X
                                                          hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                          i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                          a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                          ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                          b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                                          hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                          s : CategoryTheory.Limits.PushoutCocone a b
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                                        -/
                  _ = f ≫ a ≫ PushoutCocone.inl s := by rw [PushoutCocone.condition]
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                    /-
                                                      C : Type u_1
                                                      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                      inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                      inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                      inst✝² : CategoryTheory.NormalEpiCategory C
                                                      X Y Z : C
                                                      a : Quiver.Hom X Y
                                                      b : Quiver.Hom X Z
                                                      inst✝¹ : CategoryTheory.Epi a
                                                      inst✝ : CategoryTheory.Epi b
                                                      P : C
                                                      f : Quiver.Hom P X
                                                      hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                      i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                      Q : C
                                                      g : Quiver.Hom Q X
                                                      hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                      i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                      a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                      ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                      b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                                      hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                      s : CategoryTheory.Limits.PushoutCocone a b
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                                    -/
                  _ = 0 ≫ PushoutCocone.inl s := by rw [← Category.assoc, eq_whisker hfa]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                           /-
                                             C : Type u_1
                                             inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                             inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                             inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                             inst✝³ : CategoryTheory.Limits.HasCokernels C
                                             inst✝² : CategoryTheory.NormalEpiCategory C
                                             X Y Z : C
                                             a : Quiver.Hom X Y
                                             b : Quiver.Hom X Z
                                             inst✝¹ : CategoryTheory.Epi a
                                             inst✝ : CategoryTheory.Epi b
                                             P : C
                                             f : Quiver.Hom P X
                                             hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                             i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                             Q : C
                                             g : Quiver.Hom Q X
                                             hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                             i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                             a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                             ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                             b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                             hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                             s : CategoryTheory.Limits.PushoutCocone a b
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 s.inl) (CategoryTheory.CategoryStru …
                                           -/
                  _ = coprod.inl ≫ 0 := by rw [comp_zero, zero_comp]
                                           /-
                                             🎉 no goals
                                           -/
                  )
                (calc
                  coprod.inr ≫ coprod.desc f g ≫ b ≫ PushoutCocone.inr s =
                                                        /-
                                                          C : Type u_1
                                                          inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                          inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                          inst✝² : CategoryTheory.NormalEpiCategory C
                                                          X Y Z : C
                                                          a : Quiver.Hom X Y
                                                          b : Quiver.Hom X Z
                                                          inst✝¹ : CategoryTheory.Epi a
                                                          inst✝ : CategoryTheory.Epi b
                                                          P : C
                                                          f : Quiver.Hom P X
                                                          hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                          i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                          Q : C
                                                          g : Quiver.Hom Q X
                                                          hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                          i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                          a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                          ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                          b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                                          hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                          s : CategoryTheory.Limits.PushoutCocone a b
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
                                                        -/
                      g ≫ b ≫ PushoutCocone.inr s := by rw [coprod.inr_desc_assoc]
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                    /-
                                                      C : Type u_1
                                                      inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                      inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                      inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                      inst✝² : CategoryTheory.NormalEpiCategory C
                                                      X Y Z : C
                                                      a : Quiver.Hom X Y
                                                      b : Quiver.Hom X Z
                                                      inst✝¹ : CategoryTheory.Epi a
                                                      inst✝ : CategoryTheory.Epi b
                                                      P : C
                                                      f : Quiver.Hom P X
                                                      hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                      i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                      Q : C
                                                      g : Quiver.Hom Q X
                                                      hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                      i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                      a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                      ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                      b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                                      hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                      s : CategoryTheory.Limits.PushoutCocone a b
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
                                                    -/
                  _ = 0 ≫ PushoutCocone.inr s := by rw [← Category.assoc, eq_whisker hgb]
                                                    /-
                                                      🎉 no goals
                                                    -/
                                           /-
                                             C : Type u_1
                                             inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                             inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                             inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                             inst✝³ : CategoryTheory.Limits.HasCokernels C
                                             inst✝² : CategoryTheory.NormalEpiCategory C
                                             X Y Z : C
                                             a : Quiver.Hom X Y
                                             b : Quiver.Hom X Z
                                             inst✝¹ : CategoryTheory.Epi a
                                             inst✝ : CategoryTheory.Epi b
                                             P : C
                                             f : Quiver.Hom P X
                                             hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                             i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                             Q : C
                                             g : Quiver.Hom Q X
                                             hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                             i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                             a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                             ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                             b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                             hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                             s : CategoryTheory.Limits.PushoutCocone a b
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 s.inr) (CategoryTheory.CategoryStru …
                                           -/
                  _ = coprod.inr ≫ 0 := by rw [comp_zero, zero_comp]
                                           /-
                                             🎉 no goals
                                           -/
                  ))
          (fun s =>
            (cancel_epi a).1 <| by
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                inst✝³ : CategoryTheory.Limits.HasCokernels C
                inst✝² : CategoryTheory.NormalEpiCategory C
                X Y Z : C
                a : Quiver.Hom X Y
                b : Quiver.Hom X Z
                inst✝¹ : CategoryTheory.Epi a
                inst✝ : CategoryTheory.Epi b
                P : C
                f : Quiver.Hom P X
                hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                Q : C
                g : Quiver.Hom Q X
                hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                s : CategoryTheory.Limits.PushoutCocone a b
                ⊢ Eq (CategoryTheory.CategoryStruct.comp a (CategoryTheory.CategoryStruct.comp …
              -/
              rw [CokernelCofork.π_ofπ] at ha'
              have reassoced {W : C} (h : cokernel (coprod.desc f g) ⟶ W) : a ≫ a' ≫ h
                = cokernel.π (coprod.desc f g) ≫ h := by rw [← Category.assoc, eq_whisker ha']
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                inst✝³ : CategoryTheory.Limits.HasCokernels C
                inst✝² : CategoryTheory.NormalEpiCategory C
                X Y Z : C
                a : Quiver.Hom X Y
                b : Quiver.Hom X Z
                inst✝¹ : CategoryTheory.Epi a
                inst✝ : CategoryTheory.Epi b
                P : C
                f : Quiver.Hom P X
                hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                Q : C
                g : Quiver.Hom Q X
                hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                ha' : Eq (CategoryTheory.CategoryStruct.comp a a') (CategoryTheory.Limits.coke …
                b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                s : CategoryTheory.Limits.PushoutCocone a b
                reassoced : ∀ {W : C} (h : Quiver.Hom (CategoryTheory.Limits.cokernel (Categor …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp a (CategoryTheory.CategoryStruct.comp …
              -/
              simp [reassoced , PushoutCocone.condition s])
              /-
                🎉 no goals
              -/
          (fun s =>
            (cancel_epi b).1 <| by
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                inst✝³ : CategoryTheory.Limits.HasCokernels C
                inst✝² : CategoryTheory.NormalEpiCategory C
                X Y Z : C
                a : Quiver.Hom X Y
                b : Quiver.Hom X Z
                inst✝¹ : CategoryTheory.Epi a
                inst✝ : CategoryTheory.Epi b
                P : C
                f : Quiver.Hom P X
                hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                Q : C
                g : Quiver.Hom Q X
                hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                s : CategoryTheory.Limits.PushoutCocone a b
                ⊢ Eq (CategoryTheory.CategoryStruct.comp b (CategoryTheory.CategoryStruct.comp …
              -/
              rw [CokernelCofork.π_ofπ] at hb'
              have reassoced' {W : C} (h : cokernel (coprod.desc f g) ⟶ W) : b ≫ b' ≫ h
                = cokernel.π (coprod.desc f g) ≫ h := by rw [← Category.assoc, eq_whisker hb']
              /-
                C : Type u_1
                inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                inst✝³ : CategoryTheory.Limits.HasCokernels C
                inst✝² : CategoryTheory.NormalEpiCategory C
                X Y Z : C
                a : Quiver.Hom X Y
                b : Quiver.Hom X Z
                inst✝¹ : CategoryTheory.Epi a
                inst✝ : CategoryTheory.Epi b
                P : C
                f : Quiver.Hom P X
                hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                Q : C
                g : Quiver.Hom Q X
                hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                hb' : Eq (CategoryTheory.CategoryStruct.comp b b') (CategoryTheory.Limits.coke …
                s : CategoryTheory.Limits.PushoutCocone a b
                reassoced' : ∀ {W : C} (h : Quiver.Hom (CategoryTheory.Limits.cokernel (Catego …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp b (CategoryTheory.CategoryStruct.comp …
              -/
              simp [reassoced'])
              /-
                🎉 no goals
              -/
          fun s m h₁ _ =>
          (cancel_epi (cokernel.π (coprod.desc f g))).1 <|
            calc
              cokernel.π (coprod.desc f g) ≫ m = (a ≫ a') ≫ m := by
                /-
                  C : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                  inst✝³ : CategoryTheory.Limits.HasCokernels C
                  inst✝² : CategoryTheory.NormalEpiCategory C
                  X Y Z : C
                  a : Quiver.Hom X Y
                  b : Quiver.Hom X Z
                  inst✝¹ : CategoryTheory.Epi a
                  inst✝ : CategoryTheory.Epi b
                  P : C
                  f : Quiver.Hom P X
                  hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                  i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                  Q : C
                  g : Quiver.Hom Q X
                  hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                  i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                  a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                  ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                  b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                  hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                  s : CategoryTheory.Limits.PushoutCocone a b
                  m : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.coprod.d …
                  h₁ : Eq (CategoryTheory.CategoryStruct.comp a' m) s.inl
                  x✝ : Eq (CategoryTheory.CategoryStruct.comp b' m) s.inr
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π (Ca …
                -/
                congr
                /-
                  case e_a
                  C : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                  inst✝³ : CategoryTheory.Limits.HasCokernels C
                  inst✝² : CategoryTheory.NormalEpiCategory C
                  X Y Z : C
                  a : Quiver.Hom X Y
                  b : Quiver.Hom X Z
                  inst✝¹ : CategoryTheory.Epi a
                  inst✝ : CategoryTheory.Epi b
                  P : C
                  f : Quiver.Hom P X
                  hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                  i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                  Q : C
                  g : Quiver.Hom Q X
                  hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                  i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                  a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                  ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                  b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                  hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                  s : CategoryTheory.Limits.PushoutCocone a b
                  m : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.coprod.d …
                  h₁ : Eq (CategoryTheory.CategoryStruct.comp a' m) s.inl
                  x✝ : Eq (CategoryTheory.CategoryStruct.comp b' m) s.inr
                  ⊢ Eq (CategoryTheory.Limits.cokernel.π (CategoryTheory.Limits.coprod.desc f g) …
                -/
                exact ha'.symm
                /-
                  🎉 no goals
                -/
                                                /-
                                                  C : Type u_1
                                                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                                                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                                                  inst✝³ : CategoryTheory.Limits.HasCokernels C
                                                  inst✝² : CategoryTheory.NormalEpiCategory C
                                                  X Y Z : C
                                                  a : Quiver.Hom X Y
                                                  b : Quiver.Hom X Z
                                                  inst✝¹ : CategoryTheory.Epi a
                                                  inst✝ : CategoryTheory.Epi b
                                                  P : C
                                                  f : Quiver.Hom P X
                                                  hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                                                  i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                                  Q : C
                                                  g : Quiver.Hom Q X
                                                  hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                                                  i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                                                  a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                                                  ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                  b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                                                  hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                                                  s : CategoryTheory.Limits.PushoutCocone a b
                                                  m : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.coprod.d …
                                                  h₁ : Eq (CategoryTheory.CategoryStruct.comp a' m) s.inl
                                                  x✝ : Eq (CategoryTheory.CategoryStruct.comp b' m) s.inr
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp a …
                                                -/
              _ = a ≫ PushoutCocone.inl s := by rw [Category.assoc, h₁]
                                                /-
                                                  🎉 no goals
                                                -/
              _ = b ≫ PushoutCocone.inr s := PushoutCocone.condition s
              _ =
                  cokernel.π (coprod.desc f g) ≫
                    cokernel.desc (coprod.desc f g) (b ≫ PushoutCocone.inr s) _ := by
                /-
                  C : Type u_1
                  inst✝⁶ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝⁴ : CategoryTheory.Limits.HasFiniteCoproducts C
                  inst✝³ : CategoryTheory.Limits.HasCokernels C
                  inst✝² : CategoryTheory.NormalEpiCategory C
                  X Y Z : C
                  a : Quiver.Hom X Y
                  b : Quiver.Hom X Z
                  inst✝¹ : CategoryTheory.Epi a
                  inst✝ : CategoryTheory.Epi b
                  P : C
                  f : Quiver.Hom P X
                  hfa : Eq (CategoryTheory.CategoryStruct.comp f a) 0
                  i : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                  Q : C
                  g : Quiver.Hom Q X
                  hgb : Eq (CategoryTheory.CategoryStruct.comp g b) 0
                  i' : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
                  a' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ a hfa).pt (CategoryT …
                  ha' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                  b' : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ b hgb).pt (CategoryT …
                  hb' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π ( …
                  s : CategoryTheory.Limits.PushoutCocone a b
                  m : Quiver.Hom (CategoryTheory.Limits.cokernel (CategoryTheory.Limits.coprod.d …
                  h₁ : Eq (CategoryTheory.CategoryStruct.comp a' m) s.inl
                  x✝ : Eq (CategoryTheory.CategoryStruct.comp b' m) s.inr
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp b s.inr) (CategoryTheory.CategoryStru …
                -/
                rw [cokernel.π_desc]
                /-
                  🎉 no goals
                -/
               }


/-- The pushout of `(𝟙 Y, f)` and `(𝟙 Y, g)`. -/
private abbrev Q {X Y : C} (f g : X ⟶ Y) [Epi (coprod.desc (𝟙 Y) f)] [Epi (coprod.desc (𝟙 Y) g)] :
    C :=
  pushout (coprod.desc (𝟙 Y) f) (coprod.desc (𝟙 Y) g)


/-- The coequalizer of `f` and `g` exists. -/
@[irreducible, nolint defLemma] -- Porting note: changed to def and restored irreducible
def hasColimit_parallelPair {X Y : C} (f g : X ⟶ Y) : HasColimit (parallelPair f g) :=
  have huv : (pushout.inl _ _ : Y ⟶ Q f g) = pushout.inr _ _ :=
    calc
      (pushout.inl _ _ : Y ⟶ Q f g) = 𝟙 _ ≫ pushout.inl _ _ := Eq.symm <| Category.id_comp _
                                                                     /-
                                                                       C : Type u_1
                                                                       inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                       inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                                                       inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                                                       inst✝ : CategoryTheory.NormalEpiCategory C
                                                                       X Y : C
                                                                       f g : Quiver.Hom X Y
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y)  …
                                                                     -/
      _ = (coprod.inl ≫ coprod.desc (𝟙 Y) f) ≫ pushout.inl _ _ := by rw [coprod.inl_desc]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
      _ = (coprod.inl ≫ coprod.desc (𝟙 Y) g) ≫ pushout.inr _ _ := by
        /-
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          inst✝ : CategoryTheory.NormalEpiCategory C
          X Y : C
          f g : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
        -/
        simp only [Category.assoc, pushout.condition]
        /-
          🎉 no goals
        -/
                                /-
                                  C : Type u_1
                                  inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                  inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                  inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                  inst✝ : CategoryTheory.NormalEpiCategory C
                                  X Y : C
                                  f g : Quiver.Hom X Y
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
                                -/
      _ = pushout.inr _ _ := by rw [coprod.inl_desc, Category.id_comp]
                                /-
                                  🎉 no goals
                                -/

  have hvu : f ≫ (pushout.inl _ _ : Y ⟶ Q f g) = g ≫ pushout.inr _ _ :=
    calc
      f ≫ (pushout.inl _ _ : Y ⟶ Q f g) = (coprod.inr ≫ coprod.desc (𝟙 Y) f) ≫ pushout.inl _ _ := by
        /-
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          inst✝ : CategoryTheory.NormalEpiCategory C
          X Y : C
          f g : Quiver.Hom X Y
          huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout.inl  …
        -/
        rw [coprod.inr_desc]
        /-
          🎉 no goals
        -/
      _ = (coprod.inr ≫ coprod.desc (𝟙 Y) g) ≫ pushout.inr _ _ := by
        /-
          C : Type u_1
          inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          inst✝ : CategoryTheory.NormalEpiCategory C
          X Y : C
          f g : Quiver.Hom X Y
          huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
        -/
        simp only [Category.assoc, pushout.condition]
        /-
          🎉 no goals
        -/
                                    /-
                                      C : Type u_1
                                      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                      inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                      inst✝ : CategoryTheory.NormalEpiCategory C
                                      X Y : C
                                      f g : Quiver.Hom X Y
                                      huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp C …
                                    -/
      _ = g ≫ pushout.inr _ _ := by rw [coprod.inr_desc]
                                    /-
                                      🎉 no goals
                                    -/

                                                                           /-
                                                                             C : Type u_1
                                                                             inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                                                             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                             inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                                                             inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                                                             inst✝ : CategoryTheory.NormalEpiCategory C
                                                                             X Y : C
                                                                             f g : Quiver.Hom X Y
                                                                             huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                                                                             hvu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout.inl  …
                                                                           -/
  have huu : f ≫ (pushout.inl _ _ : Y ⟶ Q f g) = g ≫ pushout.inl _ _ := by rw [hvu, huv]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  HasColimit.mk
    { cocone := Cofork.ofπ (pushout.inl _ _) huu
      isColimit :=
        Cofork.IsColimit.mk _
          (fun s =>
            pushout.desc (Cofork.π s) (Cofork.π s) <|
                                 /-
                                   C : Type u_1
                                   inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                   inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                   inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                   inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                   inst✝ : CategoryTheory.NormalEpiCategory C
                                   X Y : C
                                   f g : Quiver.Hom X Y
                                   huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                                   hvu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                                   huu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                                   s : CategoryTheory.Limits.Cofork f g
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inl (Cat …
                                 -/
              coprod.hom_ext (by simp only [coprod.inl_desc_assoc])
                                 /-
                                   🎉 no goals
                                 -/
                    /-
                      C : Type u_1
                      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                      inst✝¹ : CategoryTheory.Limits.HasCokernels C
                      inst✝ : CategoryTheory.NormalEpiCategory C
                      X Y : C
                      f g : Quiver.Hom X Y
                      huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                      hvu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                      huu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                      s : CategoryTheory.Limits.Cofork f g
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.coprod.inr (Cat …
                    -/
                (by simp only [coprod.desc_comp, Cofork.condition s]))
                    /-
                      🎉 no goals
                    -/
                       /-
                         C : Type u_1
                         inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                         inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                         inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                         inst✝¹ : CategoryTheory.Limits.HasCokernels C
                         inst✝ : CategoryTheory.NormalEpiCategory C
                         X Y : C
                         f g : Quiver.Hom X Y
                         huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                         hvu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                         huu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                         s : CategoryTheory.Limits.Cofork f g
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ (Ca …
                       -/
          (fun s => by simp only [pushout.inl_desc, Cofork.π_ofπ]) fun s m h =>
                       /-
                         🎉 no goals
                       -/
                              /-
                                C : Type u_1
                                inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                inst✝ : CategoryTheory.NormalEpiCategory C
                                X Y : C
                                f g : Quiver.Hom X Y
                                huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                                hvu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                                huu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                                s : CategoryTheory.Limits.Cofork f g
                                m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ (CategoryTheory.Limits.pushou …
                                h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ ( …
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl (C …
                              -/
          pushout.hom_ext (by simpa only [pushout.inl_desc] using h)
                              /-
                                🎉 no goals
                              -/
                /-
                  C : Type u_1
                  inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                  inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                  inst✝¹ : CategoryTheory.Limits.HasCokernels C
                  inst✝ : CategoryTheory.NormalEpiCategory C
                  X Y : C
                  f g : Quiver.Hom X Y
                  huv : Eq (CategoryTheory.Limits.pushout.inl (CategoryTheory.Limits.coprod.desc …
                  hvu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                  huu : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.pushout. …
                  s : CategoryTheory.Limits.Cofork f g
                  m : Quiver.Hom (CategoryTheory.Limits.Cofork.ofπ (CategoryTheory.Limits.pushou …
                  h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.ofπ ( …
                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr (C …
                -/
            (by simpa only [huv.symm, pushout.inl_desc] using h) }
                /-
                  🎉 no goals
                -/


/-- A `NormalEpiCategory` category with finite coproducts and cokernels has all coequalizers. -/
instance (priority := 100) hasCoequalizers : HasCoequalizers C :=
  hasCoequalizers_of_hasColimit_parallelPair _


/-- If a zero morphism is a kernel of `f`, then `f` is a monomorphism. -/
theorem mono_of_zero_kernel {X Y : C} (f : X ⟶ Y) (Z : C)
                                                                /-
                                                                  C : Type u_1
                                                                  inst✝⁴ : CategoryTheory.Category.{?u.307869, u_1} C
                                                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                  inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
                                                                  inst✝¹ : CategoryTheory.Limits.HasCokernels C
                                                                  inst✝ : CategoryTheory.NormalEpiCategory C
                                                                  X Y : C
                                                                  f : Quiver.Hom X Y
                                                                  Z : C
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
                                                                -/
    (l : IsLimit (KernelFork.ofι (0 : Z ⟶ X) (show 0 ≫ f = 0 by simp))) : Mono f :=
                                                                /-
                                                                  🎉 no goals
                                                                -/
  ⟨fun u v huv => by
    /-
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      ⊢ Eq u v
    -/
    obtain ⟨W, w, hw, hl⟩ := normalEpiOfEpi (coequalizer.π u v)
    /-
      case mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom W X
      hw : Eq (CategoryTheory.CategoryStruct.comp w (CategoryTheory.Limits.coequaliz …
      hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      ⊢ Eq u v
    -/
    obtain ⟨m, hm⟩ := coequalizer.desc' f huv
    have reassoced {W : C} (h : coequalizer u v ⟶ W) : w ≫ coequalizer.π u v ≫ h = 0 ≫ h := by
      rw [← Category.assoc, eq_whisker hw]
    /-
      case mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom W X
      hw : Eq (CategoryTheory.CategoryStruct.comp w (CategoryTheory.Limits.coequaliz …
      hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      m : Quiver.Hom (CategoryTheory.Limits.coequalizer u v) Y
      hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer …
      reassoced : ∀ {W_1 : C} (h : Quiver.Hom (CategoryTheory.Limits.coequalizer u v …
      ⊢ Eq u v
    -/
    have hwf : w ≫ f = 0 := by rw [← hm, reassoced, zero_comp]
    /-
      case mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom W X
      hw : Eq (CategoryTheory.CategoryStruct.comp w (CategoryTheory.Limits.coequaliz …
      hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      m : Quiver.Hom (CategoryTheory.Limits.coequalizer u v) Y
      hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer …
      reassoced : ∀ {W_1 : C} (h : Quiver.Hom (CategoryTheory.Limits.coequalizer u v …
      hwf : Eq (CategoryTheory.CategoryStruct.comp w f) 0
      ⊢ Eq u v
    -/
    obtain ⟨n, hn⟩ := KernelFork.IsLimit.lift' l _ hwf
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom W X
      hw : Eq (CategoryTheory.CategoryStruct.comp w (CategoryTheory.Limits.coequaliz …
      hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      m : Quiver.Hom (CategoryTheory.Limits.coequalizer u v) Y
      hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer …
      reassoced : ∀ {W_1 : C} (h : Quiver.Hom (CategoryTheory.Limits.coequalizer u v …
      hwf : Eq (CategoryTheory.CategoryStruct.comp w f) 0
      n : Quiver.Hom W (CategoryTheory.Limits.KernelFork.ofι 0 ⋯).pt
      hn : Eq (CategoryTheory.CategoryStruct.comp n (CategoryTheory.Limits.Fork.ι (C …
      ⊢ Eq u v
    -/
    rw [Fork.ι_ofι, HasZeroMorphisms.comp_zero] at hn
    have : IsIso (coequalizer.π u v) := by
      apply isIso_colimit_cocone_parallelPair_of_eq hn.symm hl
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom W X
      hw : Eq (CategoryTheory.CategoryStruct.comp w (CategoryTheory.Limits.coequaliz …
      hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      m : Quiver.Hom (CategoryTheory.Limits.coequalizer u v) Y
      hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer …
      reassoced : ∀ {W_1 : C} (h : Quiver.Hom (CategoryTheory.Limits.coequalizer u v …
      hwf : Eq (CategoryTheory.CategoryStruct.comp w f) 0
      n : Quiver.Hom W (CategoryTheory.Limits.KernelFork.ofι 0 ⋯).pt
      hn : Eq 0 w
      this : CategoryTheory.IsIso (CategoryTheory.Limits.coequalizer.π u v)
      ⊢ Eq u v
    -/
    apply (cancel_mono (coequalizer.π u v)).1
    /-
      case mk.mk.mk
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      inst✝² : CategoryTheory.Limits.HasFiniteCoproducts C
      inst✝¹ : CategoryTheory.Limits.HasCokernels C
      inst✝ : CategoryTheory.NormalEpiCategory C
      X Y : C
      f : Quiver.Hom X Y
      Z : C
      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι 0 ⋯)
      Z✝ : C
      u v : Quiver.Hom Z✝ X
      huv : Eq (CategoryTheory.CategoryStruct.comp u f) (CategoryTheory.CategoryStru …
      W : C
      w : Quiver.Hom W X
      hw : Eq (CategoryTheory.CategoryStruct.comp w (CategoryTheory.Limits.coequaliz …
      hl : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
      m : Quiver.Hom (CategoryTheory.Limits.coequalizer u v) Y
      hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer …
      reassoced : ∀ {W_1 : C} (h : Quiver.Hom (CategoryTheory.Limits.coequalizer u v …
      hwf : Eq (CategoryTheory.CategoryStruct.comp w f) 0
      n : Quiver.Hom W (CategoryTheory.Limits.KernelFork.ofι 0 ⋯).pt
      hn : Eq 0 w
      this : CategoryTheory.IsIso (CategoryTheory.Limits.coequalizer.π u v)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Limits.coequalizer. …
    -/
    exact coequalizer.condition _ _⟩
    /-
      🎉 no goals
    -/


/-- If `g ≫ f = 0` implies `g = 0` for all `g`, then `f` is a monomorphism. -/
theorem mono_of_cancel_zero {X Y : C} (f : X ⟶ Y)
    (hf : ∀ (Z : C) (g : Z ⟶ X) (_ : g ≫ f = 0), g = 0) : Mono f :=
  mono_of_zero_kernel f 0 <| zeroKernelOfCancelZero f hf


