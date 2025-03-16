/-- A morphism `f` has a kernel if the functor `ParallelPair f 0` has a limit. -/
abbrev HasKernel {X Y : C} (f : X ⟶ Y) : Prop :=
  HasLimit (parallelPair f 0)


/-- A morphism `f` has a cokernel if the functor `ParallelPair f 0` has a colimit. -/
abbrev HasCokernel {X Y : C} (f : X ⟶ Y) : Prop :=
  HasColimit (parallelPair f 0)


/-- A kernel fork is just a fork where the second morphism is a zero morphism. -/
abbrev KernelFork :=
  Fork f 0


@[reassoc (attr := simp)]
theorem KernelFork.condition (s : KernelFork f) : Fork.ι s ≫ f = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    s : CategoryTheory.Limits.KernelFork f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι s) f) 0
  -/
  rw [Fork.condition, HasZeroMorphisms.comp_zero]
  /-
    🎉 no goals
  -/


theorem KernelFork.app_one (s : KernelFork f) : s.π.app one = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    s : CategoryTheory.Limits.KernelFork f
    ⊢ Eq (s.π.app CategoryTheory.Limits.WalkingParallelPair.one) 0
  -/
  simp [Fork.app_one_eq_ι_comp_right]
  /-
    🎉 no goals
  -/


/-- A morphism `ι` satisfying `ι ≫ f = 0` determines a kernel fork over `f`. -/
abbrev KernelFork.ofι {Z : C} (ι : Z ⟶ X) (w : ι ≫ f = 0) : KernelFork f :=
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     X Y : C
                     f : Quiver.Hom X Y
                     Z : C
                     ι : Quiver.Hom Z X
                     w : Eq (CategoryTheory.CategoryStruct.comp ι f) 0
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruct.c …
                   -/
  Fork.ofι ι <| by rw [w, HasZeroMorphisms.comp_zero]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem KernelFork.ι_ofι {X Y P : C} (f : X ⟶ Y) (ι : P ⟶ X) (w : ι ≫ f = 0) :
    Fork.ι (KernelFork.ofι ι w) = ι := rfl


/-- Every kernel fork `s` is isomorphic (actually, equal) to `fork.ofι (fork.ι s) _`. -/
def isoOfι (s : Fork f 0) : s ≅ Fork.ofι (Fork.ι s) (Fork.condition s) :=
                               /-
                                 C : Type u
                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                 X Y : C
                                 f : Quiver.Hom X Y
                                 s : CategoryTheory.Limits.Fork f 0
                                 ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq (s.π.app j) (CategoryT …
                               -/
                                              /-
                                                🎉 no goals
                                              -/
  Cones.ext (Iso.refl _) <| by rintro ⟨j⟩ <;> simp
                                              /-
                                                🎉 no goals
                                              -/


/-- If `ι = ι'`, then `fork.ofι ι _` and `fork.ofι ι' _` are isomorphic. -/
def ofιCongr {P : C} {ι ι' : P ⟶ X} {w : ι ≫ f = 0} (h : ι = ι') :
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                 X Y : C
                                                 f : Quiver.Hom X Y
                                                 P : C
                                                 ι ι' : Quiver.Hom P X
                                                 w : Eq (CategoryTheory.CategoryStruct.comp ι f) 0
                                                 h : Eq ι ι'
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ι' f) 0
                                               -/
    KernelFork.ofι ι w ≅ KernelFork.ofι ι' (by rw [← h, w]) :=
                                               /-
                                                 🎉 no goals
                                               -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    P : C
    ι ι' : Quiver.Hom P X
    w : Eq (CategoryTheory.CategoryStruct.comp ι f) 0
    h : Eq ι ι'
    ⊢ ∀ (j : CategoryTheory.Limits.WalkingParallelPair), Eq ((CategoryTheory.Limit …
  -/
  Cones.ext (Iso.refl _)
  /-
    🎉 no goals
  -/


/-- If `F` is an equivalence, then applying `F` to a diagram indexing a (co)kernel of `f` yields
    the diagram indexing the (co)kernel of `F.map f`. -/
def compNatIso {D : Type u'} [Category.{v} D] [HasZeroMorphisms D] (F : C ⥤ D) [F.IsEquivalence] :
    parallelPair f 0 ⋙ F ≅ parallelPair (F.map f) 0 :=
  let app (j : WalkingParallelPair) :
      (parallelPair f 0 ⋙ F).obj j ≅ (parallelPair (F.map f) 0).obj j :=
    match j with
    | zero => Iso.refl _
    | one => Iso.refl _
                                /-
                                  C : Type u
                                  inst✝⁴ : CategoryTheory.Category.{v, u} C
                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                  X Y : C
                                  f : Quiver.Hom X Y
                                  D : Type u'
                                  inst✝² : CategoryTheory.Category.{v, u'} D
                                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                  F : CategoryTheory.Functor C D
                                  inst✝ : F.IsEquivalence
                                  app : (j : CategoryTheory.Limits.WalkingParallelPair) → CategoryTheory.Iso ((( …
                                  ⊢ ∀ {X_1 Y_1 : CategoryTheory.Limits.WalkingParallelPair} (f_1 : Quiver.Hom X_ …
                                -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  NatIso.ofComponents app <| by rintro ⟨i⟩ ⟨j⟩ <;> intro g <;> cases g <;> simp [app]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- If `s` is a limit kernel fork and `k : W ⟶ X` satisfies `k ≫ f = 0`, then there is some
    `l : W ⟶ s.X` such that `l ≫ fork.ι s = k`. -/
def KernelFork.IsLimit.lift' {s : KernelFork f} (hs : IsLimit s) {W : C} (k : W ⟶ X)
    (h : k ≫ f = 0) : { l : W ⟶ s.pt // l ≫ Fork.ι s = k } :=
  ⟨hs.lift <| KernelFork.ofι _ h, hs.fac _ _⟩


/-- This is a slightly more convenient method to verify that a kernel fork is a limit cone. It
    only asks for a proof of facts that carry any mathematical content -/
def isLimitAux (t : KernelFork f) (lift : ∀ s : KernelFork f, s.pt ⟶ t.pt)
    (fac : ∀ s : KernelFork f, lift s ≫ t.ι = s.ι)
    (uniq : ∀ (s : KernelFork f) (m : s.pt ⟶ t.pt) (_ : m ≫ t.ι = s.ι), m = lift s) : IsLimit t :=
  { lift
    fac := fun s j => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        t : CategoryTheory.Limits.KernelFork f
        lift : (s : CategoryTheory.Limits.KernelFork f) → Quiver.Hom s.pt t.pt
        fac : ∀ (s : CategoryTheory.Limits.KernelFork f), Eq (CategoryTheory.CategoryS …
        uniq : ∀ (s : CategoryTheory.Limits.KernelFork f) (m : Quiver.Hom s.pt t.pt),  …
        s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f 0)
        j : CategoryTheory.Limits.WalkingParallelPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app j)) (s.π.app j)
      -/
      cases j
        /-
          case zero
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          t : CategoryTheory.Limits.KernelFork f
          lift : (s : CategoryTheory.Limits.KernelFork f) → Quiver.Hom s.pt t.pt
          fac : ∀ (s : CategoryTheory.Limits.KernelFork f), Eq (CategoryTheory.CategoryS …
          uniq : ∀ (s : CategoryTheory.Limits.KernelFork f) (m : Quiver.Hom s.pt t.pt),  …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app CategoryTheory.Limi …
        -/
      · exact fac s
        /-
          🎉 no goals
        -/
        /-
          case one
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          t : CategoryTheory.Limits.KernelFork f
          lift : (s : CategoryTheory.Limits.KernelFork f) → Quiver.Hom s.pt t.pt
          fac : ∀ (s : CategoryTheory.Limits.KernelFork f), Eq (CategoryTheory.CategoryS …
          uniq : ∀ (s : CategoryTheory.Limits.KernelFork f) (m : Quiver.Hom s.pt t.pt),  …
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair f 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (lift s) (t.π.app CategoryTheory.Limi …
        -/
      · simp
        /-
          🎉 no goals
        -/
    uniq := fun s m w => uniq s m (w Limits.WalkingParallelPair.zero) }


/-- This is a more convenient formulation to show that a `KernelFork` constructed using
`KernelFork.ofι` is a limit cone.
-/
def KernelFork.IsLimit.ofι {W : C} (g : W ⟶ X) (eq : g ≫ f = 0)
    (lift : ∀ {W' : C} (g' : W' ⟶ X) (_ : g' ≫ f = 0), W' ⟶ W)
    (fac : ∀ {W' : C} (g' : W' ⟶ X) (eq' : g' ≫ f = 0), lift g' eq' ≫ g = g')
    (uniq :
      ∀ {W' : C} (g' : W' ⟶ X) (eq' : g' ≫ f = 0) (m : W' ⟶ W) (_ : m ≫ g = g'), m = lift g' eq') :
    IsLimit (KernelFork.ofι g eq) :=
  isLimitAux _ (fun s => lift s.ι s.condition) (fun s => fac s.ι s.condition) fun s =>
    uniq s.ι s.condition


/-- This is a more convenient formulation to show that a `KernelFork` of the form
`KernelFork.ofι i _` is a limit cone when we know that `i` is a monomorphism. -/
def KernelFork.IsLimit.ofι' {X Y K : C} {f : X ⟶ Y} (i : K ⟶ X) (w : i ≫ f = 0)
    (h : ∀ {A : C} (k : A ⟶ X) (_ : k ≫ f = 0), { l : A ⟶ K // l ≫ i = k}) [hi : Mono i] :
    IsLimit (KernelFork.ofι i w) :=
  ofι _ _ (fun {_} k hk => (h k hk).1) (fun {_} k hk => (h k hk).2) (fun {A} k hk m hm => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X✝ Y✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      X Y K : C
      f : Quiver.Hom X Y
      i : Quiver.Hom K X
      w : Eq (CategoryTheory.CategoryStruct.comp i f) 0
      h : {A : C} → (k : Quiver.Hom A X) → Eq (CategoryTheory.CategoryStruct.comp k  …
      hi : CategoryTheory.Mono i
      A : C
      k : Quiver.Hom A X
      hk : Eq (CategoryTheory.CategoryStruct.comp k f) 0
      m : Quiver.Hom A K
      hm : Eq (CategoryTheory.CategoryStruct.comp m i) k
      ⊢ Eq m ((fun {x} k hk => ↑(h k hk)) k hk)
    -/
    rw [← cancel_mono i, (h k hk).2, hm])
    /-
      🎉 no goals
    -/


/-- Every kernel of `f` induces a kernel of `f ≫ g` if `g` is mono. -/
def isKernelCompMono {c : KernelFork f} (i : IsLimit c) {Z} (g : Y ⟶ Z) [hg : Mono g] {h : X ⟶ Z}
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                         X Y : C
                                                         f : Quiver.Hom X Y
                                                         c : CategoryTheory.Limits.KernelFork f
                                                         i : CategoryTheory.Limits.IsLimit c
                                                         Z : C
                                                         g : Quiver.Hom Y Z
                                                         hg : CategoryTheory.Mono g
                                                         h : Quiver.Hom X Z
                                                         hh : Eq h (CategoryTheory.CategoryStruct.comp f g)
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) h) 0
                                                       -/
    (hh : h = f ≫ g) : IsLimit (KernelFork.ofι c.ι (by simp [hh]) : KernelFork h) :=
                                                       /-
                                                         🎉 no goals
                                                       -/
  Fork.IsLimit.mk' _ fun s =>
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                c : CategoryTheory.Limits.KernelFork f
                                                i : CategoryTheory.Limits.IsLimit c
                                                Z : C
                                                g : Quiver.Hom Y Z
                                                hg : CategoryTheory.Mono g
                                                h : Quiver.Hom X Z
                                                hh : Eq h (CategoryTheory.CategoryStruct.comp f g)
                                                s : CategoryTheory.Limits.Fork h 0
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι f) (CategoryTheory.CategoryStruct …
                                              -/
    let s' : KernelFork f := Fork.ofι s.ι (by rw [← cancel_mono g]; simp [← hh, s.condition])
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    let l := KernelFork.IsLimit.lift' i s'.ι s'.condition
    ⟨l.1, l.2, fun hm => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        Z : C
        g : Quiver.Hom Y Z
        hg : CategoryTheory.Mono g
        h : Quiver.Hom X Z
        hh : Eq h (CategoryTheory.CategoryStruct.comp f g)
        s : CategoryTheory.Limits.Fork h 0
        s' : CategoryTheory.Limits.KernelFork f := CategoryTheory.Limits.Fork.ofι s.ι ⋯
        l : Subtype fun l => Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory. …
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        hm : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Fork.ι ( …
        ⊢ Eq m✝ ↑l
      -/
      apply Fork.IsLimit.hom_ext i; rw [Fork.ι_ofι] at hm; rw [hm]; exact l.2.symm⟩
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem isKernelCompMono_lift {c : KernelFork f} (i : IsLimit c) {Z} (g : Y ⟶ Z) [hg : Mono g]
    {h : X ⟶ Z} (hh : h = f ≫ g) (s : KernelFork h) :
    (isKernelCompMono i g hh).lift s = i.lift (Fork.ofι s.ι (by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        Z : C
        g : Quiver.Hom Y Z
        hg : CategoryTheory.Mono g
        h : Quiver.Hom X Z
        hh : Eq h (CategoryTheory.CategoryStruct.comp f g)
        s : CategoryTheory.Limits.KernelFork h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι s) f) ( …
      -/
      rw [← cancel_mono g, Category.assoc, ← hh]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.KernelFork f
        i : CategoryTheory.Limits.IsLimit c
        Z : C
        g : Quiver.Hom Y Z
        hg : CategoryTheory.Mono g
        h : Quiver.Hom X Z
        hh : Eq h (CategoryTheory.CategoryStruct.comp f g)
        s : CategoryTheory.Limits.KernelFork h
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι s) h) ( …
      -/
      simp)) := rfl
      /-
        🎉 no goals
      -/


/-- Every kernel of `f ≫ g` is also a kernel of `f`, as long as `c.ι ≫ f` vanishes. -/
def isKernelOfComp {W : C} (g : Y ⟶ W) (h : X ⟶ W) {c : KernelFork h} (i : IsLimit c)
    (hf : c.ι ≫ f = 0) (hfg : f ≫ g = h) : IsLimit (KernelFork.ofι c.ι hf) :=
                                                             /-
                                                               C : Type u
                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                               inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               X Y : C
                                                               f : Quiver.Hom X Y
                                                               W : C
                                                               g : Quiver.Hom Y W
                                                               h : Quiver.Hom X W
                                                               c : CategoryTheory.Limits.KernelFork h
                                                               i : CategoryTheory.Limits.IsLimit c
                                                               hf : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) f …
                                                               hfg : Eq (CategoryTheory.CategoryStruct.comp f g) h
                                                               s : CategoryTheory.Limits.Fork f 0
                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι h) 0
                                                             -/
  Fork.IsLimit.mk _ (fun s => i.lift (KernelFork.ofι s.ι (by simp [← hfg])))
                                                             /-
                                                               🎉 no goals
                                                             -/
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   X Y : C
                   f : Quiver.Hom X Y
                   W : C
                   g : Quiver.Hom Y W
                   h : Quiver.Hom X W
                   c : CategoryTheory.Limits.KernelFork h
                   i : CategoryTheory.Limits.IsLimit c
                   hf : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) f …
                   hfg : Eq (CategoryTheory.CategoryStruct.comp f g) h
                   s : CategoryTheory.Limits.Fork f 0
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => i.lift (CategoryTheory.Lim …
                 -/
    (fun s => by simp only [KernelFork.ι_ofι, Fork.IsLimit.lift_ι]) fun s m h => by
                 /-
                   🎉 no goals
                 -/
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      W : C
      g : Quiver.Hom Y W
      h✝ : Quiver.Hom X W
      c : CategoryTheory.Limits.KernelFork h✝
      i : CategoryTheory.Limits.IsLimit c
      hf : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) f …
      hfg : Eq (CategoryTheory.CategoryStruct.comp f g) h✝
      s : CategoryTheory.Limits.Fork f 0
      m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limi …
      h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
      ⊢ Eq m ((fun s => i.lift (CategoryTheory.Limits.KernelFork.ofι s.ι ⋯)) s)
    -/
    apply Fork.IsLimit.hom_ext i; simpa using h
                                  /-
                                    🎉 no goals
                                  -/


/-- `X` identifies to the kernel of a zero map `X ⟶ Y`. -/
def KernelFork.IsLimit.ofId {X Y : C} (f : X ⟶ Y) (hf : f = 0) :
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                         X✝ Y✝ : C
                                                         f✝ : Quiver.Hom X✝ Y✝
                                                         X Y : C
                                                         f : Quiver.Hom X Y
                                                         hf : Eq f 0
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                                       -/
    IsLimit (KernelFork.ofι (𝟙 X) (show 𝟙 X ≫ f = 0 by rw [hf, comp_zero])) :=
                                                       /-
                                                         🎉 no goals
                                                       -/
  KernelFork.IsLimit.ofι _ _ (fun x _ => x) (fun _ _ => Category.comp_id _)
                        /-
                          C : Type u
                          inst✝¹ : CategoryTheory.Category.{v, u} C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          X✝ Y✝ : C
                          f✝ : Quiver.Hom X✝ Y✝
                          X Y : C
                          f : Quiver.Hom X Y
                          hf : Eq f 0
                          W'✝ : C
                          x✝² : Quiver.Hom W'✝ X
                          x✝¹ : Eq (CategoryTheory.CategoryStruct.comp x✝² f) 0
                          x✝ : Quiver.Hom W'✝ X
                          hb : Eq (CategoryTheory.CategoryStruct.comp x✝ (CategoryTheory.CategoryStruct. …
                          ⊢ Eq x✝ ((fun {W'} x x_1 => x) x✝² x✝¹)
                        -/
    (fun _ _ _ hb => by simp only [← hb, Category.comp_id])
                        /-
                          🎉 no goals
                        -/


/-- Any zero object identifies to the kernel of a given monomorphisms. -/
def KernelFork.IsLimit.ofMonoOfIsZero {X Y : C} {f : X ⟶ Y} (c : KernelFork f)
    (hf : Mono f) (h : IsZero c.pt) : IsLimit c :=
                                         /-
                                           C : Type u
                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                           X✝ Y✝ : C
                                           f✝ : Quiver.Hom X✝ Y✝
                                           X Y : C
                                           f : Quiver.Hom X Y
                                           c : CategoryTheory.Limits.KernelFork f
                                           hf : CategoryTheory.Mono f
                                           h : CategoryTheory.Limits.IsZero c.pt
                                           s : CategoryTheory.Limits.KernelFork f
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => 0) s) (CategoryTheory.Limi …
                                         -/
  isLimitAux _ (fun _ => 0) (fun s => by rw [zero_comp, ← cancel_mono f, zero_comp, s.condition])
                                         /-
                                           🎉 no goals
                                         -/
    (fun _ _ _ => h.eq_of_tgt _ _)


lemma KernelFork.IsLimit.isIso_ι {X Y : C} {f : X ⟶ Y} (c : KernelFork f)
    (hc : IsLimit c) (hf : f = 0) : IsIso c.ι := by
  let e : c.pt ≅ X := IsLimit.conePointUniqueUpToIso hc
    (KernelFork.IsLimit.ofId (f : X ⟶ Y) hf)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    hc : CategoryTheory.Limits.IsLimit c
    hf : Eq f 0
    e : CategoryTheory.Iso c.pt X := hc.conePointUniqueUpToIso (CategoryTheory.Lim …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.Fork.ι c)
  -/
  have eq : e.inv ≫ c.ι = 𝟙 X := Fork.IsLimit.lift_ι hc
  haveI : IsIso (e.inv ≫ c.ι) := by
    rw [eq]
    infer_instance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    hc : CategoryTheory.Limits.IsLimit c
    hf : Eq f 0
    e : CategoryTheory.Iso c.pt X := hc.conePointUniqueUpToIso (CategoryTheory.Lim …
    eq : Eq (CategoryTheory.CategoryStruct.comp e.inv (CategoryTheory.Limits.Fork. …
    this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp e.inv (Categor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.Fork.ι c)
  -/
  exact IsIso.of_isIso_comp_left e.inv c.ι
  /-
    🎉 no goals
  -/


/-- If `c` is a limit kernel fork for `g : X ⟶ Y`, `e : X ≅ X'` and `g' : X' ⟶ Y` is a morphism,
then there is a limit kernel fork for `g'` with the same point as `c` if for any
morphism `φ : W ⟶ X`, there is an equivalence `φ ≫ g = 0 ↔ φ ≫ e.hom ≫ g' = 0`. -/
def KernelFork.isLimitOfIsLimitOfIff {X Y : C} {g : X ⟶ Y} {c : KernelFork g} (hc : IsLimit c)
    {X' Y' : C} (g' : X' ⟶ Y') (e : X ≅ X')
    (iff : ∀ ⦃W : C⦄ (φ : W ⟶ X), φ ≫ g = 0 ↔ φ ≫ e.hom ≫ g' = 0) :
                                                        /-
                                                          C : Type u
                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          X✝ Y✝ : C
                                                          f : Quiver.Hom X✝ Y✝
                                                          X Y : C
                                                          g : Quiver.Hom X Y
                                                          c : CategoryTheory.Limits.KernelFork g
                                                          hc : CategoryTheory.Limits.IsLimit c
                                                          X' Y' : C
                                                          g' : Quiver.Hom X' Y'
                                                          e : CategoryTheory.Iso X X'
                                                          iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                        -/
    IsLimit (KernelFork.ofι (f := g') (c.ι ≫ e.hom) (by simp [← iff])) :=
                                                        /-
                                                          🎉 no goals
                                                        -/
  KernelFork.IsLimit.ofι _ _
    (fun s hs ↦ hc.lift (KernelFork.ofι (ι := s ≫ e.inv)
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            X Y : C
            g : Quiver.Hom X Y
            c : CategoryTheory.Limits.KernelFork g
            hc : CategoryTheory.Limits.IsLimit c
            X' Y' : C
            g' : Quiver.Hom X' Y'
            e : CategoryTheory.Iso X X'
            iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
            W'✝ : C
            s : Quiver.Hom W'✝ X'
            hs : Eq (CategoryTheory.CategoryStruct.comp s g') 0
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp s …
          -/
      (by rw [iff, Category.assoc, Iso.inv_hom_id_assoc, hs])))
          /-
            🎉 no goals
          -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     X✝ Y✝ : C
                     f : Quiver.Hom X✝ Y✝
                     X Y : C
                     g : Quiver.Hom X Y
                     c : CategoryTheory.Limits.KernelFork g
                     hc : CategoryTheory.Limits.IsLimit c
                     X' Y' : C
                     g' : Quiver.Hom X' Y'
                     e : CategoryTheory.Iso X X'
                     iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
                     W'✝ : C
                     s : Quiver.Hom W'✝ X'
                     hs : Eq (CategoryTheory.CategoryStruct.comp s g') 0
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {W'} s hs => hc.lift (CategoryT …
                   -/
    (fun s hs ↦ by simp [← cancel_mono e.inv])
                   /-
                     🎉 no goals
                   -/
                                                 /-
                                                   C : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                   X✝ Y✝ : C
                                                   f : Quiver.Hom X✝ Y✝
                                                   X Y : C
                                                   g : Quiver.Hom X Y
                                                   c : CategoryTheory.Limits.KernelFork g
                                                   hc : CategoryTheory.Limits.IsLimit c
                                                   X' Y' : C
                                                   g' : Quiver.Hom X' Y'
                                                   e : CategoryTheory.Iso X X'
                                                   iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                   W'✝ : C
                                                   s : Quiver.Hom W'✝ X'
                                                   hs : Eq (CategoryTheory.CategoryStruct.comp s g') 0
                                                   m : Quiver.Hom W'✝ (((CategoryTheory.Functor.const CategoryTheory.Limits.Walki …
                                                   hm : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.CategoryStruct.c …
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι c)) ( …
                                                 -/
    (fun s hs m hm ↦ Fork.IsLimit.hom_ext hc (by simpa [← cancel_mono e.hom] using hm))
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- If `c` is a limit kernel fork for `g : X ⟶ Y`, and `g' : X ⟶ Y'` is a another morphism,
then there is a limit kernel fork for `g'` with the same point as `c` if for any
morphism `φ : W ⟶ X`, there is an equivalence `φ ≫ g = 0 ↔ φ ≫ g' = 0`. -/
def KernelFork.isLimitOfIsLimitOfIff' {X Y : C} {g : X ⟶ Y} {c : KernelFork g} (hc : IsLimit c)
    {Y' : C} (g' : X ⟶ Y')
    (iff : ∀ ⦃W : C⦄ (φ : W ⟶ X), φ ≫ g = 0 ↔ φ ≫ g' = 0) :
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                X✝ Y✝ : C
                                                f : Quiver.Hom X✝ Y✝
                                                X Y : C
                                                g : Quiver.Hom X Y
                                                c : CategoryTheory.Limits.KernelFork g
                                                hc : CategoryTheory.Limits.IsLimit c
                                                Y' : C
                                                g' : Quiver.Hom X Y'
                                                iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) g') 0
                                              -/
    IsLimit (KernelFork.ofι (f := g') c.ι (by simp [← iff])) :=
                                              /-
                                                🎉 no goals
                                              -/
                                                                   /-
                                                                     C : Type u
                                                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                     X✝ Y✝ : C
                                                                     f : Quiver.Hom X✝ Y✝
                                                                     X Y : C
                                                                     g : Quiver.Hom X Y
                                                                     c : CategoryTheory.Limits.KernelFork g
                                                                     hc : CategoryTheory.Limits.IsLimit c
                                                                     Y' : C
                                                                     g' : Quiver.Hom X Y'
                                                                     iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                                     ⊢ ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.comp  …
                                                                   -/
  IsLimit.ofIsoLimit (isLimitOfIsLimitOfIff hc g' (Iso.refl _) (by simpa using iff))
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
     /-
       C : Type u
       inst✝¹ : CategoryTheory.Category.{v, u} C
       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
       X✝ Y✝ : C
       f : Quiver.Hom X✝ Y✝
       X Y : C
       g : Quiver.Hom X Y
       c : CategoryTheory.Limits.KernelFork g
       hc : CategoryTheory.Limits.IsLimit c
       Y' : C
       g' : Quiver.Hom X Y'
       iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom W X), Iff (Eq (CategoryTheory.CategoryStruct.c …
       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
     -/
    (Fork.ext (Iso.refl _))
     /-
       🎉 no goals
     -/


/-- The morphism between points of kernel forks induced by a morphism
in the category of arrows. -/
def mapOfIsLimit (kf : KernelFork f) {kf' : KernelFork f'} (hf' : IsLimit kf')
    (φ : Arrow.mk f ⟶ Arrow.mk f') : kf.pt ⟶ kf'.pt :=
                                               /-
                                                 C : Type u
                                                 inst✝¹ : CategoryTheory.Category.{v, u} C
                                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                 X Y : C
                                                 f : Quiver.Hom X Y
                                                 X' Y' : C
                                                 f' : Quiver.Hom X' Y'
                                                 kf : CategoryTheory.Limits.KernelFork f
                                                 kf' : CategoryTheory.Limits.KernelFork f'
                                                 hf' : CategoryTheory.Limits.IsLimit kf'
                                                 φ : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                               -/
  hf'.lift (KernelFork.ofι (kf.ι ≫ φ.left) (by simp))
                                               /-
                                                 🎉 no goals
                                               -/


@[reassoc (attr := simp)]
lemma mapOfIsLimit_ι (kf : KernelFork f) {kf' : KernelFork f'} (hf' : IsLimit kf')
    (φ : Arrow.mk f ⟶ Arrow.mk f') :
    kf.mapOfIsLimit hf' φ ≫ kf'.ι = kf.ι ≫ φ.left :=
  hf'.fac _ _


/-- The isomorphism between points of limit kernel forks induced by an isomorphism
in the category of arrows. -/
@[simps]
def mapIsoOfIsLimit {kf : KernelFork f} {kf' : KernelFork f'}
    (hf : IsLimit kf) (hf' : IsLimit kf')
    (φ : Arrow.mk f ≅ Arrow.mk f') : kf.pt ≅ kf'.pt where
  hom := kf.mapOfIsLimit hf' φ.hom
  inv := kf'.mapOfIsLimit hf φ.inv
                                            /-
                                              C : Type u
                                              inst✝¹ : CategoryTheory.Category.{v, u} C
                                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                              X Y : C
                                              f : Quiver.Hom X Y
                                              X' Y' : C
                                              f' : Quiver.Hom X' Y'
                                              kf : CategoryTheory.Limits.KernelFork f
                                              kf' : CategoryTheory.Limits.KernelFork f'
                                              hf : CategoryTheory.Limits.IsLimit kf
                                              hf' : CategoryTheory.Limits.IsLimit kf'
                                              φ : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                            -/
  hom_inv_id := Fork.IsLimit.hom_ext hf (by simp)
                                            /-
                                              🎉 no goals
                                            -/
                                             /-
                                               C : Type u
                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                               inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                               X Y : C
                                               f : Quiver.Hom X Y
                                               X' Y' : C
                                               f' : Quiver.Hom X' Y'
                                               kf : CategoryTheory.Limits.KernelFork f
                                               kf' : CategoryTheory.Limits.KernelFork f'
                                               hf : CategoryTheory.Limits.IsLimit kf
                                               hf' : CategoryTheory.Limits.IsLimit kf'
                                               φ : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                             -/
  inv_hom_id := Fork.IsLimit.hom_ext hf' (by simp)
                                             /-
                                               🎉 no goals
                                             -/


/-- The kernel of a morphism, expressed as the equalizer with the 0 morphism. -/
abbrev kernel (f : X ⟶ Y) [HasKernel f] : C :=
  equalizer f 0


/-- The map from `kernel f` into the source of `f`. -/
abbrev kernel.ι : kernel f ⟶ X :=
  equalizer.ι f 0


@[simp]
theorem equalizer_as_kernel : equalizer.ι f 0 = kernel.ι f := rfl


@[reassoc (attr := simp)]
theorem kernel.condition : kernel.ι f ≫ f = 0 :=
  KernelFork.condition _


/-- The kernel built from `kernel.ι f` is limiting. -/
def kernelIsKernel : IsLimit (Fork.ofι (kernel.ι f) ((kernel.condition f).trans comp_zero.symm)) :=
                                                                  /-
                                                                    C : Type u
                                                                    inst✝² : CategoryTheory.Category.{v, u} C
                                                                    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                    X Y : C
                                                                    f : Quiver.Hom X Y
                                                                    inst✝ : CategoryTheory.Limits.HasKernel f
                                                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
                                                                  -/
  IsLimit.ofIsoLimit (limit.isLimit _) (Fork.ext (Iso.refl _) (by aesop_cat))
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Given any morphism `k : W ⟶ X` satisfying `k ≫ f = 0`, `k` factors through `kernel.ι f`
    via `kernel.lift : W ⟶ kernel f`. -/
abbrev kernel.lift {W : C} (k : W ⟶ X) (h : k ≫ f = 0) : W ⟶ kernel f :=
  (kernelIsKernel f).lift (KernelFork.ofι k h)


@[reassoc (attr := simp)]
theorem kernel.lift_ι {W : C} (k : W ⟶ X) (h : k ≫ f = 0) : kernel.lift f k h ≫ kernel.ι f = k :=
  (kernelIsKernel f).fac (KernelFork.ofι k h) WalkingParallelPair.zero


@[simp]
theorem kernel.lift_zero {W : C} {h} : kernel.lift f (0 : W ⟶ X) h = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    W : C
    h : Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
    ⊢ Eq (CategoryTheory.Limits.kernel.lift f 0 h) 0
  -/
  ext; simp
       /-
         🎉 no goals
       -/


instance kernel.lift_mono {W : C} (k : W ⟶ X) (h : k ≫ f = 0) [Mono k] : Mono (kernel.lift f k h) :=
  ⟨fun {Z} g g' w => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasKernel f
      W : C
      k : Quiver.Hom W X
      h : Eq (CategoryTheory.CategoryStruct.comp k f) 0
      inst✝ : CategoryTheory.Mono k
      Z : C
      g g' : Quiver.Hom Z W
      w : Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.kernel.lif …
      ⊢ Eq g g'
    -/
    replace w := w =≫ kernel.ι f
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasKernel f
      W : C
      k : Quiver.Hom W X
      h : Eq (CategoryTheory.CategoryStruct.comp k f) 0
      inst✝ : CategoryTheory.Mono k
      Z : C
      g g' : Quiver.Hom Z W
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
      ⊢ Eq g g'
    -/
    simp only [Category.assoc, kernel.lift_ι] at w
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasKernel f
      W : C
      k : Quiver.Hom W X
      h : Eq (CategoryTheory.CategoryStruct.comp k f) 0
      inst✝ : CategoryTheory.Mono k
      Z : C
      g g' : Quiver.Hom Z W
      w : Eq (CategoryTheory.CategoryStruct.comp g k) (CategoryTheory.CategoryStruct …
      ⊢ Eq g g'
    -/
    exact (cancel_mono k).1 w⟩
    /-
      🎉 no goals
    -/


/-- Any morphism `k : W ⟶ X` satisfying `k ≫ f = 0` induces a morphism `l : W ⟶ kernel f` such that
    `l ≫ kernel.ι f = k`. -/
def kernel.lift' {W : C} (k : W ⟶ X) (h : k ≫ f = 0) : { l : W ⟶ kernel f // l ≫ kernel.ι f = k } :=
  ⟨kernel.lift f k h, kernel.lift_ι _ _ _⟩


/-- A commuting square induces a morphism of kernels. -/
abbrev kernel.map {X' Y' : C} (f' : X' ⟶ Y') [HasKernel f'] (p : X ⟶ X') (q : Y ⟶ Y')
    (w : f ≫ q = p ≫ f') : kernel f ⟶ kernel f' :=
                                      /-
                                        C : Type u
                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                        X Y : C
                                        f : Quiver.Hom X Y
                                        inst✝¹ : CategoryTheory.Limits.HasKernel f
                                        X' Y' : C
                                        f' : Quiver.Hom X' Y'
                                        inst✝ : CategoryTheory.Limits.HasKernel f'
                                        p : Quiver.Hom X X'
                                        q : Quiver.Hom Y Y'
                                        w : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStruct …
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                      -/
  kernel.lift f' (kernel.ι f ≫ p) (by simp [← w])
                                      /-
                                        🎉 no goals
                                      -/


/-- Given a commutative diagram
    X --f--> Y --g--> Z
    |        |        |
    |        |        |
    v        v        v
    X' -f'-> Y' -g'-> Z'
with horizontal arrows composing to zero,
then we obtain a commutative square
   X ---> kernel g
   |         |
   |         | kernel.map
   |         |
   v         v
   X' --> kernel g'
-/
theorem kernel.lift_map {X Y Z X' Y' Z' : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasKernel g] (w : f ≫ g = 0)
    (f' : X' ⟶ Y') (g' : Y' ⟶ Z') [HasKernel g'] (w' : f' ≫ g' = 0) (p : X ⟶ X') (q : Y ⟶ Y')
    (r : Z ⟶ Z') (h₁ : f ≫ q = p ≫ f') (h₂ : g ≫ r = q ≫ g') :
    kernel.lift g f w ≫ kernel.map g g' q r h₂ = p ≫ kernel.lift g' f' w' := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z X' Y' Z' : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasKernel g
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    f' : Quiver.Hom X' Y'
    g' : Quiver.Hom Y' Z'
    inst✝ : CategoryTheory.Limits.HasKernel g'
    w' : Eq (CategoryTheory.CategoryStruct.comp f' g') 0
    p : Quiver.Hom X X'
    q : Quiver.Hom Y Y'
    r : Quiver.Hom Z Z'
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp g r) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift g  …
  -/
  ext; simp [h₁]
       /-
         🎉 no goals
       -/


/-- A commuting square of isomorphisms induces an isomorphism of kernels. -/
@[simps]
def kernel.mapIso {X' Y' : C} (f' : X' ⟶ Y') [HasKernel f'] (p : X ≅ X') (q : Y ≅ Y')
    (w : f ≫ q.hom = p.hom ≫ f') : kernel f ≅ kernel f' where
  hom := kernel.map f f' p.hom q.hom w
  inv :=
    kernel.map f' f p.inv q.inv
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          X' Y' : C
          f' : Quiver.Hom X' Y'
          inst✝ : CategoryTheory.Limits.HasKernel f'
          p : CategoryTheory.Iso X X'
          q : CategoryTheory.Iso Y Y'
          w : Eq (CategoryTheory.CategoryStruct.comp f q.hom) (CategoryTheory.CategorySt …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f' q.inv) (CategoryTheory.CategoryStr …
        -/
        refine (cancel_mono q.hom).1 ?_
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          X' Y' : C
          f' : Quiver.Hom X' Y'
          inst✝ : CategoryTheory.Limits.HasKernel f'
          p : CategoryTheory.Iso X X'
          q : CategoryTheory.Iso Y Y'
          w : Eq (CategoryTheory.CategoryStruct.comp f q.hom) (CategoryTheory.CategorySt …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        simp [w])
        /-
          🎉 no goals
        -/


/-- Every kernel of the zero morphism is an isomorphism -/
instance kernel.ι_zero_isIso : IsIso (kernel.ι (0 : X ⟶ Y)) :=
  equalizer.ι_of_self _


theorem eq_zero_of_epi_kernel [Epi (kernel.ι f)] : f = 0 :=
                                  /-
                                    C : Type u
                                    inst✝³ : CategoryTheory.Category.{v, u} C
                                    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                    X Y : C
                                    f : Quiver.Hom X Y
                                    inst✝¹ : CategoryTheory.Limits.HasKernel f
                                    inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.kernel.ι f)
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) f) …
                                  -/
  (cancel_epi (kernel.ι f)).1 (by simp)
                                  /-
                                    🎉 no goals
                                  -/


/-- The kernel of a zero morphism is isomorphic to the source. -/
def kernelZeroIsoSource : kernel (0 : X ⟶ Y) ≅ X :=
  equalizer.isoSourceOfSelf 0


@[simp]
theorem kernelZeroIsoSource_hom : kernelZeroIsoSource.hom = kernel.ι (0 : X ⟶ Y) := rfl


@[simp]
theorem kernelZeroIsoSource_inv :
                                                                /-
                                                                  C : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                  X Y : C
                                                                  f : Quiver.Hom X Y
                                                                  inst✝ : CategoryTheory.Limits.HasKernel f
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X)  …
                                                                -/
    kernelZeroIsoSource.inv = kernel.lift (0 : X ⟶ Y) (𝟙 X) (by simp) := by
                                                                /-
                                                                  🎉 no goals
                                                                -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    ⊢ Eq CategoryTheory.Limits.kernelZeroIsoSource.inv (CategoryTheory.Limits.kern …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.kernelZeroIsoSo …
  -/
  simp [kernelZeroIsoSource]
  /-
    🎉 no goals
  -/


/-- If two morphisms are known to be equal, then their kernels are isomorphic. -/
def kernelIsoOfEq {f g : X ⟶ Y} [HasKernel f] [HasKernel g] (h : f = g) : kernel f ≅ kernel g :=
                           /-
                             C : Type u
                             inst✝⁴ : CategoryTheory.Category.{v, u} C
                             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                             X Y : C
                             f✝ : Quiver.Hom X Y
                             inst✝² : CategoryTheory.Limits.HasKernel f✝
                             f g : Quiver.Hom X Y
                             inst✝¹ : CategoryTheory.Limits.HasKernel f
                             inst✝ : CategoryTheory.Limits.HasKernel g
                             h : Eq f g
                             ⊢ CategoryTheory.Iso (CategoryTheory.Limits.parallelPair f 0) (CategoryTheory. …
                           -/
  HasLimit.isoOfNatIso (by rw [h])
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem kernelIsoOfEq_refl {h : f = f} : kernelIsoOfEq h = Iso.refl (kernel f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    h : Eq f f
    ⊢ Eq (CategoryTheory.Limits.kernelIsoOfEq h) (CategoryTheory.Iso.refl (Categor …
  -/
  ext
  /-
    case w.h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    h : Eq f f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelIsoOfEq  …
  -/
  simp [kernelIsoOfEq]
  /-
    🎉 no goals
  -/

/- Porting note: induction on Eq is trying instantiate another g... -/

@[reassoc (attr := simp)]
theorem kernelIsoOfEq_hom_comp_ι {f g : X ⟶ Y} [HasKernel f] [HasKernel g] (h : f = g) :
    (kernelIsoOfEq h).hom ≫ kernel.ι g = kernel.ι f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel g
    h : Eq f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelIsoOfEq  …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[reassoc (attr := simp)]
theorem kernelIsoOfEq_inv_comp_ι {f g : X ⟶ Y} [HasKernel f] [HasKernel g] (h : f = g) :
    (kernelIsoOfEq h).inv ≫ kernel.ι _ = kernel.ι _ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel g
    h : Eq f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelIsoOfEq  …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[reassoc (attr := simp)]
theorem lift_comp_kernelIsoOfEq_hom {Z} {f g : X ⟶ Y} [HasKernel f] [HasKernel g] (h : f = g)
    (e : Z ⟶ X) (he) :
                                                                     /-
                                                                       C : Type u
                                                                       inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                       inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       X Y : C
                                                                       f✝ : Quiver.Hom X Y
                                                                       inst✝² : CategoryTheory.Limits.HasKernel f✝
                                                                       Z : C
                                                                       f g : Quiver.Hom X Y
                                                                       inst✝¹ : CategoryTheory.Limits.HasKernel f
                                                                       inst✝ : CategoryTheory.Limits.HasKernel g
                                                                       h : Eq f g
                                                                       e : Quiver.Hom Z X
                                                                       he : Eq (CategoryTheory.CategoryStruct.comp e f) 0
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp e g) 0
                                                                     -/
    kernel.lift _ e he ≫ (kernelIsoOfEq h).hom = kernel.lift _ e (by simp [← h, he]) := by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel g
    h : Eq f g
    e : Quiver.Hom Z X
    he : Eq (CategoryTheory.CategoryStruct.comp e f) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift f  …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[reassoc (attr := simp)]
theorem lift_comp_kernelIsoOfEq_inv {Z} {f g : X ⟶ Y} [HasKernel f] [HasKernel g] (h : f = g)
    (e : Z ⟶ X) (he) :
                                                                     /-
                                                                       C : Type u
                                                                       inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                       inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       X Y : C
                                                                       f✝ : Quiver.Hom X Y
                                                                       inst✝² : CategoryTheory.Limits.HasKernel f✝
                                                                       Z : C
                                                                       f g : Quiver.Hom X Y
                                                                       inst✝¹ : CategoryTheory.Limits.HasKernel f
                                                                       inst✝ : CategoryTheory.Limits.HasKernel g
                                                                       h : Eq f g
                                                                       e : Quiver.Hom Z X
                                                                       he : Eq (CategoryTheory.CategoryStruct.comp e g) 0
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp e f) 0
                                                                     -/
    kernel.lift _ e he ≫ (kernelIsoOfEq h).inv = kernel.lift _ e (by simp [h, he]) := by
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel g
    h : Eq f g
    e : Quiver.Hom Z X
    he : Eq (CategoryTheory.CategoryStruct.comp e g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift g  …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[simp]
theorem kernelIsoOfEq_trans {f g h : X ⟶ Y} [HasKernel f] [HasKernel g] [HasKernel h] (w₁ : f = g)
    (w₂ : g = h) : kernelIsoOfEq w₁ ≪≫ kernelIsoOfEq w₂ = kernelIsoOfEq (w₁.trans w₂) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f g h : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasKernel f
    inst✝¹ : CategoryTheory.Limits.HasKernel g
    inst✝ : CategoryTheory.Limits.HasKernel h
    w₁ : Eq f g
    w₂ : Eq g h
    ⊢ Eq ((CategoryTheory.Limits.kernelIsoOfEq w₁).trans (CategoryTheory.Limits.ke …
  -/
  cases w₁; cases w₂; ext; simp [kernelIsoOfEq]
                           /-
                             🎉 no goals
                           -/


theorem kernel_not_epi_of_nonzero (w : f ≠ 0) : ¬Epi (kernel.ι f) := fun _ =>
  w (eq_zero_of_epi_kernel f)


theorem kernel_not_iso_of_nonzero (w : f ≠ 0) : IsIso (kernel.ι f) → False := fun _ =>
  kernel_not_epi_of_nonzero w inferInstance


instance hasKernel_comp_mono {X Y Z : C} (f : X ⟶ Y) [HasKernel f] (g : Y ⟶ Z) [Mono g] :
    HasKernel (f ≫ g) :=
  ⟨⟨{   cone := _
        isLimit := isKernelCompMono (limit.isLimit _) g rfl }⟩⟩


/-- When `g` is a monomorphism, the kernel of `f ≫ g` is isomorphic to the kernel of `f`.
-/
@[simps]
def kernelCompMono {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasKernel f] [Mono g] :
    kernel (f ≫ g) ≅ kernel f where
  hom :=
    kernel.lift _ (kernel.ι _)
      (by
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          inst✝² : CategoryTheory.Limits.HasKernel f✝
          X Y Z : C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Mono g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (Cate …
        -/
        rw [← cancel_mono g]
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          inst✝² : CategoryTheory.Limits.HasKernel f✝
          X Y Z : C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Mono g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp)
        /-
          🎉 no goals
        -/
                                        /-
                                          C : Type u
                                          inst✝⁴ : CategoryTheory.Category.{v, u} C
                                          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                          X✝ Y✝ : C
                                          f✝ : Quiver.Hom X✝ Y✝
                                          inst✝² : CategoryTheory.Limits.HasKernel f✝
                                          X Y Z : C
                                          f : Quiver.Hom X Y
                                          g : Quiver.Hom Y Z
                                          inst✝¹ : CategoryTheory.Limits.HasKernel f
                                          inst✝ : CategoryTheory.Mono g
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) (C …
                                        -/
  inv := kernel.lift _ (kernel.ι _) (by simp)
                                        /-
                                          🎉 no goals
                                        -/


instance hasKernel_iso_comp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] [HasKernel g] :
    HasKernel (f ≫ g) where
  exists_limit :=
                                                        /-
                                                          C : Type u
                                                          inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          X✝ Y✝ : C
                                                          f✝ : Quiver.Hom X✝ Y✝
                                                          inst✝² : CategoryTheory.Limits.HasKernel f✝
                                                          X Y Z : C
                                                          f : Quiver.Hom X Y
                                                          g : Quiver.Hom Y Z
                                                          inst✝¹ : CategoryTheory.IsIso f
                                                          inst✝ : CategoryTheory.Limits.HasKernel g
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                        -/
    ⟨{  cone := KernelFork.ofι (kernel.ι g ≫ inv f) (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                                      /-
                                                                        C : Type u
                                                                        inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                        X✝ Y✝ : C
                                                                        f✝ : Quiver.Hom X✝ Y✝
                                                                        inst✝² : CategoryTheory.Limits.HasKernel f✝
                                                                        X Y Z : C
                                                                        f : Quiver.Hom X Y
                                                                        g : Quiver.Hom Y Z
                                                                        inst✝¹ : CategoryTheory.IsIso f
                                                                        inst✝ : CategoryTheory.Limits.HasKernel g
                                                                        s : CategoryTheory.Limits.KernelFork (CategoryTheory.CategoryStruct.comp f g)
                                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                                      -/
        isLimit := isLimitAux _ (fun s => kernel.lift _ (s.ι ≫ f) (by aesop_cat))
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                /-
                  C : Type u
                  inst✝⁴ : CategoryTheory.Category.{v, u} C
                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  inst✝² : CategoryTheory.Limits.HasKernel f✝
                  X Y Z : C
                  f : Quiver.Hom X Y
                  g : Quiver.Hom Y Z
                  inst✝¹ : CategoryTheory.IsIso f
                  inst✝ : CategoryTheory.Limits.HasKernel g
                  ⊢ ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.CategoryStruct.comp  …
                -/
            (by aesop_cat) fun s m w => by
                /-
                  🎉 no goals
                -/
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            X✝ Y✝ : C
            f✝ : Quiver.Hom X✝ Y✝
            inst✝² : CategoryTheory.Limits.HasKernel f✝
            X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.IsIso f
            inst✝ : CategoryTheory.Limits.HasKernel g
            s : CategoryTheory.Limits.KernelFork (CategoryTheory.CategoryStruct.comp f g)
            m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Cate …
            w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
            ⊢ Eq m ((fun s => CategoryTheory.Limits.kernel.lift g (CategoryTheory.Category …
          -/
          simp_rw [← w]
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            X✝ Y✝ : C
            f✝ : Quiver.Hom X✝ Y✝
            inst✝² : CategoryTheory.Limits.HasKernel f✝
            X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.IsIso f
            inst✝ : CategoryTheory.Limits.HasKernel g
            s : CategoryTheory.Limits.KernelFork (CategoryTheory.CategoryStruct.comp f g)
            m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Cate …
            w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
            ⊢ Eq m (CategoryTheory.Limits.kernel.lift g (CategoryTheory.CategoryStruct.com …
          -/
          symm
          /-
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            X✝ Y✝ : C
            f✝ : Quiver.Hom X✝ Y✝
            inst✝² : CategoryTheory.Limits.HasKernel f✝
            X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.IsIso f
            inst✝ : CategoryTheory.Limits.HasKernel g
            s : CategoryTheory.Limits.KernelFork (CategoryTheory.CategoryStruct.comp f g)
            m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Cate …
            w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
            ⊢ Eq (CategoryTheory.Limits.kernel.lift g (CategoryTheory.CategoryStruct.comp  …
          -/
          apply equalizer.hom_ext
          /-
            case h
            C : Type u
            inst✝⁴ : CategoryTheory.Category.{v, u} C
            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
            X✝ Y✝ : C
            f✝ : Quiver.Hom X✝ Y✝
            inst✝² : CategoryTheory.Limits.HasKernel f✝
            X Y Z : C
            f : Quiver.Hom X Y
            g : Quiver.Hom Y Z
            inst✝¹ : CategoryTheory.IsIso f
            inst✝ : CategoryTheory.Limits.HasKernel g
            s : CategoryTheory.Limits.KernelFork (CategoryTheory.CategoryStruct.comp f g)
            m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Cate …
            w : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift g  …
          -/
          simp }⟩
          /-
            🎉 no goals
          -/


/-- When `f` is an isomorphism, the kernel of `f ≫ g` is isomorphic to the kernel of `g`.
-/
@[simps]
def kernelIsIsoComp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [IsIso f] [HasKernel g] :
    kernel (f ≫ g) ≅ kernel g where
                                            /-
                                              C : Type u
                                              inst✝⁴ : CategoryTheory.Category.{v, u} C
                                              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                              X✝ Y✝ : C
                                              f✝ : Quiver.Hom X✝ Y✝
                                              inst✝² : CategoryTheory.Limits.HasKernel f✝
                                              X Y Z : C
                                              f : Quiver.Hom X Y
                                              g : Quiver.Hom Y Z
                                              inst✝¹ : CategoryTheory.IsIso f
                                              inst✝ : CategoryTheory.Limits.HasKernel g
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                            -/
  hom := kernel.lift _ (kernel.ι _ ≫ f) (by simp)
                                            /-
                                              🎉 no goals
                                            -/
                                                /-
                                                  C : Type u
                                                  inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  X✝ Y✝ : C
                                                  f✝ : Quiver.Hom X✝ Y✝
                                                  inst✝² : CategoryTheory.Limits.HasKernel f✝
                                                  X Y Z : C
                                                  f : Quiver.Hom X Y
                                                  g : Quiver.Hom Y Z
                                                  inst✝¹ : CategoryTheory.IsIso f
                                                  inst✝ : CategoryTheory.Limits.HasKernel g
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                -/
  inv := kernel.lift _ (kernel.ι _ ≫ inv f) (by simp)
                                                /-
                                                  🎉 no goals
                                                -/


/-- The morphism from the zero object determines a cone on a kernel diagram -/
def kernel.zeroKernelFork : KernelFork f where
  pt := 0
  π := { app := fun _ => 0 }


/-- The map from the zero object is a kernel of a monomorphism -/
def kernel.isLimitConeZeroCone [Mono f] : IsLimit (kernel.zeroKernelFork f) :=
  Fork.IsLimit.mk _ (fun _ => 0)
    (fun s => by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Mono f
        s : CategoryTheory.Limits.Fork f 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => 0) s) (CategoryTheory.Limi …
      -/
      rw [zero_comp]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Mono f
        s : CategoryTheory.Limits.Fork f 0
        ⊢ Eq 0 s.ι
      -/
      refine (zero_of_comp_mono f ?_).symm
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Mono f
        s : CategoryTheory.Limits.Fork f 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι f) 0
      -/
      exact KernelFork.condition _)
      /-
        🎉 no goals
      -/
    fun _ _ _ => zero_of_to_zero _


/-- The kernel of a monomorphism is isomorphic to the zero object -/
def kernel.ofMono [HasKernel f] [Mono f] : kernel f ≅ 0 :=
  Functor.mapIso (Cones.forget _) <|
    IsLimit.uniqueUpToIso (limit.isLimit (parallelPair f 0)) (kernel.isLimitConeZeroCone f)


/-- The kernel morphism of a monomorphism is a zero morphism -/
theorem kernel.ι_of_mono [HasKernel f] [Mono f] : kernel.ι f = 0 :=
  zero_of_source_iso_zero _ (kernel.ofMono f)


/-- If `g ≫ f = 0` implies `g = 0` for all `g`, then `0 : 0 ⟶ X` is a kernel of `f`. -/
def zeroKernelOfCancelZero {X Y : C} (f : X ⟶ Y)
    (hf : ∀ (Z : C) (g : Z ⟶ X) (_ : g ≫ f = 0), g = 0) :
                                                           /-
                                                             C : Type u
                                                             inst✝² : CategoryTheory.Category.{v, u} C
                                                             inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             X✝ Y✝ : C
                                                             f✝ : Quiver.Hom X✝ Y✝
                                                             inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                             X Y : C
                                                             f : Quiver.Hom X Y
                                                             hf : ∀ (Z : C) (g : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp g  …
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 f) 0
                                                           -/
    IsLimit (KernelFork.ofι (0 : 0 ⟶ X) (show 0 ≫ f = 0 by simp)) :=
                                                           /-
                                                             🎉 no goals
                                                           -/
                                              /-
                                                C : Type u
                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                X✝ Y✝ : C
                                                f✝ : Quiver.Hom X✝ Y✝
                                                inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                hf : ∀ (Z : C) (g : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp g  …
                                                s : CategoryTheory.Limits.Fork f 0
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun x => 0) s) (CategoryTheory.Limi …
                                              -/
  Fork.IsLimit.mk _ (fun _ => 0) (fun s => by rw [hf _ _ (KernelFork.condition s), zero_comp])
                                              /-
                                                🎉 no goals
                                              -/
                    /-
                      C : Type u
                      inst✝² : CategoryTheory.Category.{v, u} C
                      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                      X✝ Y✝ : C
                      f✝ : Quiver.Hom X✝ Y✝
                      inst✝ : CategoryTheory.Limits.HasZeroObject C
                      X Y : C
                      f : Quiver.Hom X Y
                      hf : ∀ (Z : C) (g : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp g  …
                      s : CategoryTheory.Limits.Fork f 0
                      m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι 0 ⋯).pt
                      x✝ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (C …
                      ⊢ Eq m ((fun x => 0) s)
                    -/
    fun s m _ => by dsimp; apply HasZeroObject.to_zero_ext
                           /-
                             🎉 no goals
                           -/


/-- If `i` is an isomorphism such that `l ≫ i.hom = f`, any kernel of `f` is a kernel of `l`. -/
def IsKernel.ofCompIso {Z : C} (l : X ⟶ Z) (i : Z ≅ Y) (h : l ≫ i.hom = f) {s : KernelFork f}
    (hs : IsLimit s) :
    IsLimit
                                                             /-
                                                               C : Type u
                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                               inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               X Y : C
                                                               f : Quiver.Hom X Y
                                                               Z : C
                                                               l : Quiver.Hom X Z
                                                               i : CategoryTheory.Iso Z Y
                                                               h : Eq (CategoryTheory.CategoryStruct.comp l i.hom) f
                                                               s : CategoryTheory.Limits.KernelFork f
                                                               hs : CategoryTheory.Limits.IsLimit s
                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι s) l) 0
                                                             -/
      (KernelFork.ofι (Fork.ι s) <| show Fork.ι s ≫ l = 0 by simp [← i.comp_inv_eq.2 h.symm]) :=
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                                         /-
                                                                           C : Type u
                                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                           X Y : C
                                                                           f : Quiver.Hom X Y
                                                                           Z : C
                                                                           l : Quiver.Hom X Z
                                                                           i : CategoryTheory.Iso Z Y
                                                                           h : Eq (CategoryTheory.CategoryStruct.comp l i.hom) f
                                                                           s✝ : CategoryTheory.Limits.KernelFork f
                                                                           hs : CategoryTheory.Limits.IsLimit s✝
                                                                           s : CategoryTheory.Limits.Fork l 0
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp s.ι f) 0
                                                                         -/
  Fork.IsLimit.mk _ (fun s => hs.lift <| KernelFork.ofι (Fork.ι s) <| by simp [← h])
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   X Y : C
                   f : Quiver.Hom X Y
                   Z : C
                   l : Quiver.Hom X Z
                   i : CategoryTheory.Iso Z Y
                   h : Eq (CategoryTheory.CategoryStruct.comp l i.hom) f
                   s✝ : CategoryTheory.Limits.KernelFork f
                   hs : CategoryTheory.Limits.IsLimit s✝
                   s : CategoryTheory.Limits.Fork l 0
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => hs.lift (CategoryTheory.Li …
                 -/
    (fun s => by simp) fun s m h => by
                 /-
                   🎉 no goals
                 -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        l : Quiver.Hom X Z
        i : CategoryTheory.Iso Z Y
        h✝ : Eq (CategoryTheory.CategoryStruct.comp l i.hom) f
        s✝ : CategoryTheory.Limits.KernelFork f
        hs : CategoryTheory.Limits.IsLimit s✝
        s : CategoryTheory.Limits.Fork l 0
        m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limi …
        h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
        ⊢ Eq m ((fun s => hs.lift (CategoryTheory.Limits.KernelFork.ofι s.ι ⋯)) s)
      -/
      apply Fork.IsLimit.hom_ext hs
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        l : Quiver.Hom X Z
        i : CategoryTheory.Iso Z Y
        h✝ : Eq (CategoryTheory.CategoryStruct.comp l i.hom) f
        s✝ : CategoryTheory.Limits.KernelFork f
        hs : CategoryTheory.Limits.IsLimit s✝
        s : CategoryTheory.Limits.Fork l 0
        m : Quiver.Hom s.pt (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limi …
        h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι s✝))  …
      -/
      simpa using h
      /-
        🎉 no goals
      -/


/-- If `i` is an isomorphism such that `l ≫ i.hom = f`, the kernel of `f` is a kernel of `l`. -/
def kernel.ofCompIso [HasKernel f] {Z : C} (l : X ⟶ Z) (i : Z ≅ Y) (h : l ≫ i.hom = f) :
    IsLimit
                                                                 /-
                                                                   C : Type u
                                                                   inst✝² : CategoryTheory.Category.{v, u} C
                                                                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                   X Y : C
                                                                   f : Quiver.Hom X Y
                                                                   inst✝ : CategoryTheory.Limits.HasKernel f
                                                                   Z : C
                                                                   l : Quiver.Hom X Z
                                                                   i : CategoryTheory.Iso Z Y
                                                                   h : Eq (CategoryTheory.CategoryStruct.comp l i.hom) f
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) l) 0
                                                                 -/
      (KernelFork.ofι (kernel.ι f) <| show kernel.ι f ≫ l = 0 by simp [← i.comp_inv_eq.2 h.symm]) :=
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  IsKernel.ofCompIso f l i h <| limit.isLimit _


/-- If `s` is any limit kernel cone over `f` and if `i` is an isomorphism such that
    `i.hom ≫ s.ι = l`, then `l` is a kernel of `f`. -/
def IsKernel.isoKernel {Z : C} (l : Z ⟶ X) {s : KernelFork f} (hs : IsLimit s) (i : Z ≅ s.pt)
                                                                                /-
                                                                                  C : Type u
                                                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                  X Y : C
                                                                                  f : Quiver.Hom X Y
                                                                                  Z : C
                                                                                  l : Quiver.Hom Z X
                                                                                  s : CategoryTheory.Limits.KernelFork f
                                                                                  hs : CategoryTheory.Limits.IsLimit s
                                                                                  i : CategoryTheory.Iso Z s.pt
                                                                                  h : Eq (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.Limits.Fork.ι …
                                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp l f) 0
                                                                                -/
    (h : i.hom ≫ Fork.ι s = l) : IsLimit (KernelFork.ofι l <| show l ≫ f = 0 by simp [← h]) :=
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  IsLimit.ofIsoLimit hs <|
    Cones.ext i.symm fun j => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        l : Quiver.Hom Z X
        s : CategoryTheory.Limits.KernelFork f
        hs : CategoryTheory.Limits.IsLimit s
        i : CategoryTheory.Iso Z s.pt
        h : Eq (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.Limits.Fork.ι …
        j : CategoryTheory.Limits.WalkingParallelPair
        ⊢ Eq (s.π.app j) (CategoryTheory.CategoryStruct.comp i.symm.hom ((CategoryTheo …
      -/
      cases j
        /-
          case zero
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          Z : C
          l : Quiver.Hom Z X
          s : CategoryTheory.Limits.KernelFork f
          hs : CategoryTheory.Limits.IsLimit s
          i : CategoryTheory.Iso Z s.pt
          h : Eq (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.Limits.Fork.ι …
          ⊢ Eq (s.π.app CategoryTheory.Limits.WalkingParallelPair.zero) (CategoryTheory. …
        -/
      · exact (Iso.eq_inv_comp i).2 h
        /-
          🎉 no goals
        -/
        /-
          case one
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          Z : C
          l : Quiver.Hom Z X
          s : CategoryTheory.Limits.KernelFork f
          hs : CategoryTheory.Limits.IsLimit s
          i : CategoryTheory.Iso Z s.pt
          h : Eq (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.Limits.Fork.ι …
          ⊢ Eq (s.π.app CategoryTheory.Limits.WalkingParallelPair.one) (CategoryTheory.C …
        -/
      · dsimp; rw [← h]; simp
                         /-
                           🎉 no goals
                         -/


/-- If `i` is an isomorphism such that `i.hom ≫ kernel.ι f = l`, then `l` is a kernel of `f`. -/
def kernel.isoKernel [HasKernel f] {Z : C} (l : Z ⟶ X) (i : Z ≅ kernel f)
    (h : i.hom ≫ kernel.ι f = l) :
                                                   /-
                                                     C : Type u
                                                     inst✝² : CategoryTheory.Category.{v, u} C
                                                     inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                     X Y : C
                                                     f : Quiver.Hom X Y
                                                     inst✝ : CategoryTheory.Limits.HasKernel f
                                                     Z : C
                                                     l : Quiver.Hom Z X
                                                     i : CategoryTheory.Iso Z (CategoryTheory.Limits.kernel f)
                                                     h : Eq (CategoryTheory.CategoryStruct.comp i.hom (CategoryTheory.Limits.kernel …
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp l f) 0
                                                   -/
    IsLimit (@KernelFork.ofι _ _ _ _ _ f _ l <| by simp [← h]) :=
                                                   /-
                                                     🎉 no goals
                                                   -/
  IsKernel.isoKernel f l (limit.isLimit _) i h


/-- The kernel morphism of a zero morphism is an isomorphism -/
theorem kernel.ι_of_zero : IsIso (kernel.ι (0 : X ⟶ Y)) :=
  equalizer.ι_of_self _


/-- A cokernel cofork is just a cofork where the second morphism is a zero morphism. -/
abbrev CokernelCofork :=
  Cofork f 0


@[reassoc (attr := simp)]
theorem CokernelCofork.condition (s : CokernelCofork f) : f ≫ s.π = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    s : CategoryTheory.Limits.CokernelCofork f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π s)) 0
  -/
  rw [Cofork.condition, zero_comp]
  /-
    🎉 no goals
  -/


theorem CokernelCofork.π_eq_zero (s : CokernelCofork f) : s.ι.app zero = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    s : CategoryTheory.Limits.CokernelCofork f
    ⊢ Eq (s.ι.app CategoryTheory.Limits.WalkingParallelPair.zero) 0
  -/
  simp [Cofork.app_zero_eq_comp_π_right]
  /-
    🎉 no goals
  -/


/-- A morphism `π` satisfying `f ≫ π = 0` determines a cokernel cofork on `f`. -/
abbrev CokernelCofork.ofπ {Z : C} (π : Y ⟶ Z) (w : f ≫ π = 0) : CokernelCofork f :=
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                       X Y : C
                       f : Quiver.Hom X Y
                       Z : C
                       π : Quiver.Hom Y Z
                       w : Eq (CategoryTheory.CategoryStruct.comp f π) 0
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct.c …
                     -/
  Cofork.ofπ π <| by rw [w, zero_comp]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem CokernelCofork.π_ofπ {X Y P : C} (f : X ⟶ Y) (π : Y ⟶ P) (w : f ≫ π = 0) :
    Cofork.π (CokernelCofork.ofπ π w) = π :=
  rfl


/-- Every cokernel cofork `s` is isomorphic (actually, equal) to `cofork.ofπ (cofork.π s) _`. -/
def isoOfπ (s : Cofork f 0) : s ≅ Cofork.ofπ (Cofork.π s) (Cofork.condition s) :=
                                       /-
                                         C : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                         X Y : C
                                         f : Quiver.Hom X Y
                                         s : CategoryTheory.Limits.Cofork f 0
                                         j : CategoryTheory.Limits.WalkingParallelPair
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) (CategoryTheory.Iso.refl  …
                                       -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  Cocones.ext (Iso.refl _) fun j => by cases j <;> aesop_cat
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If `π = π'`, then `CokernelCofork.of_π π _` and `CokernelCofork.of_π π' _` are isomorphic. -/
def ofπCongr {P : C} {π π' : Y ⟶ P} {w : f ≫ π = 0} (h : π = π') :
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                         X Y : C
                                                         f : Quiver.Hom X Y
                                                         P : C
                                                         π π' : Quiver.Hom Y P
                                                         w : Eq (CategoryTheory.CategoryStruct.comp f π) 0
                                                         h : Eq π π'
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f π') 0
                                                       -/
    CokernelCofork.ofπ π w ≅ CokernelCofork.ofπ π' (by rw [← h, w]) :=
                                                       /-
                                                         🎉 no goals
                                                       -/
                                       /-
                                         C : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                         inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                         X Y : C
                                         f : Quiver.Hom X Y
                                         P : C
                                         π π' : Quiver.Hom Y P
                                         w : Eq (CategoryTheory.CategoryStruct.comp f π) 0
                                         h : Eq π π'
                                         j : CategoryTheory.Limits.WalkingParallelPair
                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.CokernelCofor …
                                       -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  Cocones.ext (Iso.refl _) fun j => by cases j <;> aesop_cat
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- If `s` is a colimit cokernel cofork, then every `k : Y ⟶ W` satisfying `f ≫ k = 0` induces
    `l : s.X ⟶ W` such that `cofork.π s ≫ l = k`. -/
def CokernelCofork.IsColimit.desc' {s : CokernelCofork f} (hs : IsColimit s) {W : C} (k : Y ⟶ W)
    (h : f ≫ k = 0) : { l : s.pt ⟶ W // Cofork.π s ≫ l = k } :=
  ⟨hs.desc <| CokernelCofork.ofπ _ h, hs.fac _ _⟩


/-- This is a slightly more convenient method to verify that a cokernel cofork is a colimit cocone.
It only asks for a proof of facts that carry any mathematical content -/
def isColimitAux (t : CokernelCofork f) (desc : ∀ s : CokernelCofork f, t.pt ⟶ s.pt)
    (fac : ∀ s : CokernelCofork f, t.π ≫ desc s = s.π)
    (uniq : ∀ (s : CokernelCofork f) (m : t.pt ⟶ s.pt) (_ : t.π ≫ m = s.π), m = desc s) :
    IsColimit t :=
  { desc
    fac := fun s j => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        t : CategoryTheory.Limits.CokernelCofork f
        desc : (s : CategoryTheory.Limits.CokernelCofork f) → Quiver.Hom t.pt s.pt
        fac : ∀ (s : CategoryTheory.Limits.CokernelCofork f), Eq (CategoryTheory.Categ …
        uniq : ∀ (s : CategoryTheory.Limits.CokernelCofork f) (m : Quiver.Hom t.pt s.p …
        s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f 0)
        j : CategoryTheory.Limits.WalkingParallelPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app j) (desc s)) (s.ι.app j)
      -/
      cases j
        /-
          case zero
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          t : CategoryTheory.Limits.CokernelCofork f
          desc : (s : CategoryTheory.Limits.CokernelCofork f) → Quiver.Hom t.pt s.pt
          fac : ∀ (s : CategoryTheory.Limits.CokernelCofork f), Eq (CategoryTheory.Categ …
          uniq : ∀ (s : CategoryTheory.Limits.CokernelCofork f) (m : Quiver.Hom t.pt s.p …
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app CategoryTheory.Limits.Walkin …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case one
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          t : CategoryTheory.Limits.CokernelCofork f
          desc : (s : CategoryTheory.Limits.CokernelCofork f) → Quiver.Hom t.pt s.pt
          fac : ∀ (s : CategoryTheory.Limits.CokernelCofork f), Eq (CategoryTheory.Categ …
          uniq : ∀ (s : CategoryTheory.Limits.CokernelCofork f) (m : Quiver.Hom t.pt s.p …
          s : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair f 0)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (t.ι.app CategoryTheory.Limits.Walkin …
        -/
      · exact fac s
        /-
          🎉 no goals
        -/
    uniq := fun s m w => uniq s m (w Limits.WalkingParallelPair.one) }


/-- This is a more convenient formulation to show that a `CokernelCofork` constructed using
`CokernelCofork.ofπ` is a limit cone.
-/
def CokernelCofork.IsColimit.ofπ {Z : C} (g : Y ⟶ Z) (eq : f ≫ g = 0)
    (desc : ∀ {Z' : C} (g' : Y ⟶ Z') (_ : f ≫ g' = 0), Z ⟶ Z')
    (fac : ∀ {Z' : C} (g' : Y ⟶ Z') (eq' : f ≫ g' = 0), g ≫ desc g' eq' = g')
    (uniq :
      ∀ {Z' : C} (g' : Y ⟶ Z') (eq' : f ≫ g' = 0) (m : Z ⟶ Z') (_ : g ≫ m = g'), m = desc g' eq') :
    IsColimit (CokernelCofork.ofπ g eq) :=
  isColimitAux _ (fun s => desc s.π s.condition) (fun s => fac s.π s.condition) fun s =>
    uniq s.π s.condition


/-- This is a more convenient formulation to show that a `CokernelCofork` of the form
`CokernelCofork.ofπ p _` is a colimit cocone when we know that `p` is an epimorphism. -/
def CokernelCofork.IsColimit.ofπ' {X Y Q : C} {f : X ⟶ Y} (p : Y ⟶ Q) (w : f ≫ p = 0)
    (h : ∀ {A : C} (k : Y ⟶ A) (_ : f ≫ k = 0), { l : Q ⟶ A // p ≫ l = k}) [hp : Epi p] :
    IsColimit (CokernelCofork.ofπ p w) :=
  ofπ _ _ (fun {_} k hk => (h k hk).1) (fun {_} k hk => (h k hk).2) (fun {A} k hk m hm => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X✝ Y✝ : C
      f✝ : Quiver.Hom X✝ Y✝
      X Y Q : C
      f : Quiver.Hom X Y
      p : Quiver.Hom Y Q
      w : Eq (CategoryTheory.CategoryStruct.comp f p) 0
      h : {A : C} → (k : Quiver.Hom Y A) → Eq (CategoryTheory.CategoryStruct.comp f  …
      hp : CategoryTheory.Epi p
      A : C
      k : Quiver.Hom Y A
      hk : Eq (CategoryTheory.CategoryStruct.comp f k) 0
      m : Quiver.Hom Q A
      hm : Eq (CategoryTheory.CategoryStruct.comp p m) k
      ⊢ Eq m ((fun {x} k hk => ↑(h k hk)) k hk)
    -/
    rw [← cancel_epi p, (h k hk).2, hm])
    /-
      🎉 no goals
    -/


/-- Every cokernel of `f` induces a cokernel of `g ≫ f` if `g` is epi. -/
def isCokernelEpiComp {c : CokernelCofork f} (i : IsColimit c) {W} (g : W ⟶ X) [hg : Epi g]
    {h : W ⟶ Y} (hh : h = g ≫ f) :
                                          /-
                                            C : Type u
                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                            X Y : C
                                            f : Quiver.Hom X Y
                                            c : CategoryTheory.Limits.CokernelCofork f
                                            i : CategoryTheory.Limits.IsColimit c
                                            W : C
                                            g : Quiver.Hom W X
                                            hg : CategoryTheory.Epi g
                                            h : Quiver.Hom W Y
                                            hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Limits.Cofork.π c)) 0
                                          -/
    IsColimit (CokernelCofork.ofπ c.π (by rw [hh]; simp) : CokernelCofork h) :=
                                                   /-
                                                     🎉 no goals
                                                   -/
  Cofork.IsColimit.mk' _ fun s =>
    let s' : CokernelCofork f :=
      Cofork.ofπ s.π
        (by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            X Y : C
            f : Quiver.Hom X Y
            c : CategoryTheory.Limits.CokernelCofork f
            i : CategoryTheory.Limits.IsColimit c
            W : C
            g : Quiver.Hom W X
            hg : CategoryTheory.Epi g
            h : Quiver.Hom W Y
            hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
            s : CategoryTheory.Limits.Cofork h 0
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.π) (CategoryTheory.CategoryStruct …
          -/
          apply hg.left_cancellation
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            X Y : C
            f : Quiver.Hom X Y
            c : CategoryTheory.Limits.CokernelCofork f
            i : CategoryTheory.Limits.IsColimit c
            W : C
            g : Quiver.Hom W X
            hg : CategoryTheory.Epi g
            h : Quiver.Hom W Y
            hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
            s : CategoryTheory.Limits.Cofork h 0
            ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.comp …
          -/
          rw [← Category.assoc, ← hh, s.condition]
          /-
            case a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            X Y : C
            f : Quiver.Hom X Y
            c : CategoryTheory.Limits.CokernelCofork f
            i : CategoryTheory.Limits.IsColimit c
            W : C
            g : Quiver.Hom W X
            hg : CategoryTheory.Epi g
            h : Quiver.Hom W Y
            hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
            s : CategoryTheory.Limits.Cofork h 0
            ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 s.π) (CategoryTheory.CategoryStruct …
          -/
          simp)
          /-
            🎉 no goals
          -/
    let l := CokernelCofork.IsColimit.desc' i s'.π s'.condition
    ⟨l.1, l.2, fun hm => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        c : CategoryTheory.Limits.CokernelCofork f
        i : CategoryTheory.Limits.IsColimit c
        W : C
        g : Quiver.Hom W X
        hg : CategoryTheory.Epi g
        h : Quiver.Hom W Y
        hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
        s : CategoryTheory.Limits.Cofork h 0
        s' : CategoryTheory.Limits.CokernelCofork f := CategoryTheory.Limits.Cofork.of …
        l : Subtype fun l => Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
        ⊢ Eq m✝ ↑l
      -/
      apply Cofork.IsColimit.hom_ext i; rw [Cofork.π_ofπ] at hm; rw [hm]; exact l.2.symm⟩
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem isCokernelEpiComp_desc {c : CokernelCofork f} (i : IsColimit c) {W} (g : W ⟶ X) [hg : Epi g]
    {h : W ⟶ Y} (hh : h = g ≫ f) (s : CokernelCofork h) :
    (isCokernelEpiComp i g hh).desc s =
      i.desc
        (Cofork.ofπ s.π
          (by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              X Y : C
              f : Quiver.Hom X Y
              c : CategoryTheory.Limits.CokernelCofork f
              i : CategoryTheory.Limits.IsColimit c
              W : C
              g : Quiver.Hom W X
              hg : CategoryTheory.Epi g
              h : Quiver.Hom W Y
              hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
              s : CategoryTheory.Limits.CokernelCofork h
              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π s)) …
            -/
            rw [← cancel_epi g, ← Category.assoc, ← hh]
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
              X Y : C
              f : Quiver.Hom X Y
              c : CategoryTheory.Limits.CokernelCofork f
              i : CategoryTheory.Limits.IsColimit c
              W : C
              g : Quiver.Hom W X
              hg : CategoryTheory.Epi g
              h : Quiver.Hom W Y
              hh : Eq h (CategoryTheory.CategoryStruct.comp g f)
              s : CategoryTheory.Limits.CokernelCofork h
              ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.Limits.Cofork.π s)) …
            -/
            simp)) :=
            /-
              🎉 no goals
            -/
  rfl


/-- Every cokernel of `g ≫ f` is also a cokernel of `f`, as long as `f ≫ c.π` vanishes. -/
def isCokernelOfComp {W : C} (g : W ⟶ X) (h : W ⟶ Y) {c : CokernelCofork h} (i : IsColimit c)
    (hf : f ≫ c.π = 0) (hfg : g ≫ f = h) : IsColimit (CokernelCofork.ofπ c.π hf) :=
                                                                     /-
                                                                       C : Type u
                                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       X Y : C
                                                                       f : Quiver.Hom X Y
                                                                       W : C
                                                                       g : Quiver.Hom W X
                                                                       h : Quiver.Hom W Y
                                                                       c : CategoryTheory.Limits.CokernelCofork h
                                                                       i : CategoryTheory.Limits.IsColimit c
                                                                       hf : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π  …
                                                                       hfg : Eq (CategoryTheory.CategoryStruct.comp g f) h
                                                                       s : CategoryTheory.Limits.Cofork f 0
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp h s.π) 0
                                                                     -/
  Cofork.IsColimit.mk _ (fun s => i.desc (CokernelCofork.ofπ s.π (by simp [← hfg])))
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   X Y : C
                   f : Quiver.Hom X Y
                   W : C
                   g : Quiver.Hom W X
                   h : Quiver.Hom W Y
                   c : CategoryTheory.Limits.CokernelCofork h
                   i : CategoryTheory.Limits.IsColimit c
                   hf : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π  …
                   hfg : Eq (CategoryTheory.CategoryStruct.comp g f) h
                   s : CategoryTheory.Limits.Cofork f 0
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                 -/
    (fun s => by simp only [CokernelCofork.π_ofπ, Cofork.IsColimit.π_desc]) fun s m h => by
                 /-
                   🎉 no goals
                 -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        W : C
        g : Quiver.Hom W X
        h✝ : Quiver.Hom W Y
        c : CategoryTheory.Limits.CokernelCofork h✝
        i : CategoryTheory.Limits.IsColimit c
        hf : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π  …
        hfg : Eq (CategoryTheory.CategoryStruct.comp g f) h✝
        s : CategoryTheory.Limits.Cofork f 0
        m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limit …
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
        ⊢ Eq m ((fun s => i.desc (CategoryTheory.Limits.CokernelCofork.ofπ s.π ⋯)) s)
      -/
      apply Cofork.IsColimit.hom_ext i
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        W : C
        g : Quiver.Hom W X
        h✝ : Quiver.Hom W Y
        c : CategoryTheory.Limits.CokernelCofork h✝
        i : CategoryTheory.Limits.IsColimit c
        hf : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π  …
        hfg : Eq (CategoryTheory.CategoryStruct.comp g f) h✝
        s : CategoryTheory.Limits.Cofork f 0
        m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limit …
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) m) …
      -/
      simpa using h
      /-
        🎉 no goals
      -/


/-- `Y` identifies to the cokernel of a zero map `X ⟶ Y`. -/
def CokernelCofork.IsColimit.ofId {X Y : C} (f : X ⟶ Y) (hf : f = 0) :
                                                             /-
                                                               C : Type u
                                                               inst✝¹ : CategoryTheory.Category.{v, u} C
                                                               inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               X✝ Y✝ : C
                                                               f✝ : Quiver.Hom X✝ Y✝
                                                               X Y : C
                                                               f : Quiver.Hom X Y
                                                               hf : Eq f 0
                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                                                             -/
    IsColimit (CokernelCofork.ofπ (𝟙 Y) (show f ≫ 𝟙 Y = 0 by rw [hf, zero_comp])) :=
                                                             /-
                                                               🎉 no goals
                                                             -/
  CokernelCofork.IsColimit.ofπ _ _ (fun x _ => x) (fun _ _ => Category.id_comp _)
                        /-
                          C : Type u
                          inst✝¹ : CategoryTheory.Category.{v, u} C
                          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                          X✝ Y✝ : C
                          f✝ : Quiver.Hom X✝ Y✝
                          X Y : C
                          f : Quiver.Hom X Y
                          hf : Eq f 0
                          Z'✝ : C
                          x✝² : Quiver.Hom Y Z'✝
                          x✝¹ : Eq (CategoryTheory.CategoryStruct.comp f x✝²) 0
                          x✝ : Quiver.Hom Y Z'✝
                          hb : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id  …
                          ⊢ Eq x✝ ((fun {Z'} x x_1 => x) x✝² x✝¹)
                        -/
    (fun _ _ _ hb => by simp only [← hb, Category.id_comp])
                        /-
                          🎉 no goals
                        -/


/-- Any zero object identifies to the cokernel of a given epimorphisms. -/
def CokernelCofork.IsColimit.ofEpiOfIsZero {X Y : C} {f : X ⟶ Y} (c : CokernelCofork f)
    (hf : Epi f) (h : IsZero c.pt) : IsColimit c :=
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                             X✝ Y✝ : C
                                             f✝ : Quiver.Hom X✝ Y✝
                                             X Y : C
                                             f : Quiver.Hom X Y
                                             c : CategoryTheory.Limits.CokernelCofork f
                                             hf : CategoryTheory.Epi f
                                             h : CategoryTheory.Limits.IsZero c.pt
                                             s : CategoryTheory.Limits.CokernelCofork f
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) (( …
                                           -/
  isColimitAux _ (fun _ => 0) (fun s => by rw [comp_zero, ← cancel_epi f, comp_zero, s.condition])
                                           /-
                                             🎉 no goals
                                           -/
    (fun _ _ _ => h.eq_of_src _ _)


lemma CokernelCofork.IsColimit.isIso_π {X Y : C} {f : X ⟶ Y} (c : CokernelCofork f)
    (hc : IsColimit c) (hf : f = 0) : IsIso c.π := by
  let e : c.pt ≅ Y := IsColimit.coconePointUniqueUpToIso hc
    (CokernelCofork.IsColimit.ofId (f : X ⟶ Y) hf)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    hc : CategoryTheory.Limits.IsColimit c
    hf : Eq f 0
    e : CategoryTheory.Iso c.pt Y := hc.coconePointUniqueUpToIso (CategoryTheory.L …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.Cofork.π c)
  -/
  have eq : c.π ≫ e.hom = 𝟙 Y := Cofork.IsColimit.π_desc hc
  haveI : IsIso (c.π ≫ e.hom) := by
    rw [eq]
    dsimp
    infer_instance
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    hc : CategoryTheory.Limits.IsColimit c
    hf : Eq f 0
    e : CategoryTheory.Iso c.pt Y := hc.coconePointUniqueUpToIso (CategoryTheory.L …
    eq : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) …
    this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheor …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.Cofork.π c)
  -/
  exact IsIso.of_isIso_comp_right c.π e.hom
  /-
    🎉 no goals
  -/


/-- If `c` is a colimit cokernel cofork for `f : X ⟶ Y`, `e : Y ≅ Y'` and `f' : X' ⟶ Y` is a
morphism, then there is a colimit cokernel cofork for `f'` with the same point as `c` if for any
morphism `φ : Y ⟶ W`, there is an equivalence `f ≫ φ = 0 ↔ f' ≫ e.hom ≫ φ = 0`. -/
def CokernelCofork.isColimitOfIsColimitOfIff {X Y : C} {f : X ⟶ Y} {c : CokernelCofork f}
    (hc : IsColimit c) {X' Y' : C} (f' : X' ⟶ Y') (e : Y' ≅ Y)
    (iff : ∀ ⦃W : C⦄ (φ : Y ⟶ W), f ≫ φ = 0 ↔ f' ≫ e.hom ≫ φ = 0) :
                                                              /-
                                                                C : Type u
                                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                X✝ Y✝ : C
                                                                f✝ : Quiver.Hom X✝ Y✝
                                                                X Y : C
                                                                f : Quiver.Hom X Y
                                                                c : CategoryTheory.Limits.CokernelCofork f
                                                                hc : CategoryTheory.Limits.IsColimit c
                                                                X' Y' : C
                                                                f' : Quiver.Hom X' Y'
                                                                e : CategoryTheory.Iso Y' Y
                                                                iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f' (CategoryTheory.CategoryStruct.com …
                                                              -/
    IsColimit (CokernelCofork.ofπ (f := f') (e.hom ≫ c.π) (by simp [← iff])) :=
                                                              /-
                                                                🎉 no goals
                                                              -/
  CokernelCofork.IsColimit.ofπ _ _
    (fun s hs ↦ hc.desc (CokernelCofork.ofπ (π := e.inv ≫ s)
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
            X✝ Y✝ : C
            f✝ : Quiver.Hom X✝ Y✝
            X Y : C
            f : Quiver.Hom X Y
            c : CategoryTheory.Limits.CokernelCofork f
            hc : CategoryTheory.Limits.IsColimit c
            X' Y' : C
            f' : Quiver.Hom X' Y'
            e : CategoryTheory.Iso Y' Y
            iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
            Z'✝ : C
            s : Quiver.Hom Y' Z'✝
            hs : Eq (CategoryTheory.CategoryStruct.comp f' s) 0
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
          -/
      (by rw [iff, e.hom_inv_id_assoc, hs])))
          /-
            🎉 no goals
          -/
                   /-
                     C : Type u
                     inst✝¹ : CategoryTheory.Category.{v, u} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     X✝ Y✝ : C
                     f✝ : Quiver.Hom X✝ Y✝
                     X Y : C
                     f : Quiver.Hom X Y
                     c : CategoryTheory.Limits.CokernelCofork f
                     hc : CategoryTheory.Limits.IsColimit c
                     X' Y' : C
                     f' : Quiver.Hom X' Y'
                     e : CategoryTheory.Iso Y' Y
                     iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
                     Z'✝ : C
                     s : Quiver.Hom Y' Z'✝
                     hs : Eq (CategoryTheory.CategoryStruct.comp f' s) 0
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
                   -/
    (fun s hs ↦ by simp [← cancel_epi e.inv])
                   /-
                     🎉 no goals
                   -/
                                                     /-
                                                       C : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                       X✝ Y✝ : C
                                                       f✝ : Quiver.Hom X✝ Y✝
                                                       X Y : C
                                                       f : Quiver.Hom X Y
                                                       c : CategoryTheory.Limits.CokernelCofork f
                                                       hc : CategoryTheory.Limits.IsColimit c
                                                       X' Y' : C
                                                       f' : Quiver.Hom X' Y'
                                                       e : CategoryTheory.Iso Y' Y
                                                       iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                       Z'✝ : C
                                                       s : Quiver.Hom Y' Z'✝
                                                       hs : Eq (CategoryTheory.CategoryStruct.comp f' s) 0
                                                       m : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingPa …
                                                       hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) m) …
                                                     -/
    (fun s hs m hm ↦ Cofork.IsColimit.hom_ext hc (by simpa [← cancel_epi e.hom] using hm))
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- If `c` is a colimit cokernel cofork for `f : X ⟶ Y`, and `f' : X' ⟶ Y is another
morphism, then there is a colimit cokernel cofork for `f'` with the same point as `c` if for any
morphism `φ : Y ⟶ W`, there is an equivalence `f ≫ φ = 0 ↔ f' ≫ φ = 0`. -/
def CokernelCofork.isColimitOfIsColimitOfIff' {X Y : C} {f : X ⟶ Y} {c : CokernelCofork f}
    (hc : IsColimit c) {X' : C} (f' : X' ⟶ Y)
    (iff : ∀ ⦃W : C⦄ (φ : Y ⟶ W), f ≫ φ = 0 ↔ f' ≫ φ = 0) :
                                                    /-
                                                      C : Type u
                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                      X✝ Y✝ : C
                                                      f✝ : Quiver.Hom X✝ Y✝
                                                      X Y : C
                                                      f : Quiver.Hom X Y
                                                      c : CategoryTheory.Limits.CokernelCofork f
                                                      hc : CategoryTheory.Limits.IsColimit c
                                                      X' : C
                                                      f' : Quiver.Hom X' Y
                                                      iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp f' (CategoryTheory.Limits.Cofork.π c) …
                                                    -/
    IsColimit (CokernelCofork.ofπ (f := f') c.π (by simp [← iff])) :=
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                             X✝ Y✝ : C
                                                                             f✝ : Quiver.Hom X✝ Y✝
                                                                             X Y : C
                                                                             f : Quiver.Hom X Y
                                                                             c : CategoryTheory.Limits.CokernelCofork f
                                                                             hc : CategoryTheory.Limits.IsColimit c
                                                                             X' : C
                                                                             f' : Quiver.Hom X' Y
                                                                             iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
                                                                             ⊢ ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.comp  …
                                                                           -/
  IsColimit.ofIsoColimit (isColimitOfIsColimitOfIff hc f' (Iso.refl _) (by simpa using iff))
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
     /-
       C : Type u
       inst✝¹ : CategoryTheory.Category.{v, u} C
       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
       X✝ Y✝ : C
       f✝ : Quiver.Hom X✝ Y✝
       X Y : C
       f : Quiver.Hom X Y
       c : CategoryTheory.Limits.CokernelCofork f
       hc : CategoryTheory.Limits.IsColimit c
       X' : C
       f' : Quiver.Hom X' Y
       iff : ∀ ⦃W : C⦄ (φ : Quiver.Hom Y W), Iff (Eq (CategoryTheory.CategoryStruct.c …
       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
     -/
    (Cofork.ext (Iso.refl _))
     /-
       🎉 no goals
     -/


/-- The morphism between points of cokernel coforks induced by a morphism
in the category of arrows. -/
def mapOfIsColimit {cc : CokernelCofork f} (hf : IsColimit cc) (cc' : CokernelCofork f')
    (φ : Arrow.mk f ⟶ Arrow.mk f') : cc.pt ⟶ cc'.pt :=
  hf.desc (CokernelCofork.ofπ (φ.right ≫ cc'.π) (by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      X' Y' : C
      f' : Quiver.Hom X' Y'
      cc : CategoryTheory.Limits.CokernelCofork f
      hf : CategoryTheory.Limits.IsColimit cc
      cc' : CategoryTheory.Limits.CokernelCofork f'
      φ : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    erw [← Arrow.w_assoc φ, condition, comp_zero]))
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma π_mapOfIsColimit {cc : CokernelCofork f} (hf : IsColimit cc) (cc' : CokernelCofork f')
    (φ : Arrow.mk f ⟶ Arrow.mk f') :
    cc.π ≫ mapOfIsColimit hf cc' φ = φ.right ≫ cc'.π :=
  hf.fac _ _


/-- The isomorphism between points of limit cokernel coforks induced by an isomorphism
in the category of arrows. -/
@[simps]
def mapIsoOfIsColimit {cc : CokernelCofork f} {cc' : CokernelCofork f'}
    (hf : IsColimit cc) (hf' : IsColimit cc')
    (φ : Arrow.mk f ≅ Arrow.mk f') : cc.pt ≅ cc'.pt where
  hom := mapOfIsColimit hf cc' φ.hom
  inv := mapOfIsColimit hf' cc φ.inv
                                                /-
                                                  C : Type u
                                                  inst✝¹ : CategoryTheory.Category.{v, u} C
                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  X Y : C
                                                  f : Quiver.Hom X Y
                                                  X' Y' : C
                                                  f' : Quiver.Hom X' Y'
                                                  cc : CategoryTheory.Limits.CokernelCofork f
                                                  cc' : CategoryTheory.Limits.CokernelCofork f'
                                                  hf : CategoryTheory.Limits.IsColimit cc
                                                  hf' : CategoryTheory.Limits.IsColimit cc'
                                                  φ : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π cc) ( …
                                                -/
  hom_inv_id := Cofork.IsColimit.hom_ext hf (by simp)
                                                /-
                                                  🎉 no goals
                                                -/
                                                 /-
                                                   C : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                   X Y : C
                                                   f : Quiver.Hom X Y
                                                   X' Y' : C
                                                   f' : Quiver.Hom X' Y'
                                                   cc : CategoryTheory.Limits.CokernelCofork f
                                                   cc' : CategoryTheory.Limits.CokernelCofork f'
                                                   hf : CategoryTheory.Limits.IsColimit cc
                                                   hf' : CategoryTheory.Limits.IsColimit cc'
                                                   φ : CategoryTheory.Iso (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π cc')  …
                                                 -/
  inv_hom_id := Cofork.IsColimit.hom_ext hf' (by simp)
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- The cokernel of a morphism, expressed as the coequalizer with the 0 morphism. -/
abbrev cokernel : C :=
  coequalizer f 0


/-- The map from the target of `f` to `cokernel f`. -/
abbrev cokernel.π : Y ⟶ cokernel f :=
  coequalizer.π f 0


@[simp]
theorem coequalizer_as_cokernel : coequalizer.π f 0 = cokernel.π f :=
  rfl


@[reassoc (attr := simp)]
theorem cokernel.condition : f ≫ cokernel.π f = 0 :=
  CokernelCofork.condition _


/-- The cokernel built from `cokernel.π f` is colimiting. -/
def cokernelIsCokernel :
    IsColimit (Cofork.ofπ (cokernel.π f) ((cokernel.condition f).trans zero_comp.symm)) :=
                                                /-
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  X Y : C
                                                  f : Quiver.Hom X Y
                                                  inst✝ : CategoryTheory.Limits.HasCokernel f
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                                                -/
  IsColimit.ofIsoColimit (colimit.isColimit _) (Cofork.ext (Iso.refl _))
                                                /-
                                                  🎉 no goals
                                                -/


/-- Given any morphism `k : Y ⟶ W` such that `f ≫ k = 0`, `k` factors through `cokernel.π f`
    via `cokernel.desc : cokernel f ⟶ W`. -/
abbrev cokernel.desc {W : C} (k : Y ⟶ W) (h : f ≫ k = 0) : cokernel f ⟶ W :=
  (cokernelIsCokernel f).desc (CokernelCofork.ofπ k h)


@[reassoc (attr := simp)]
theorem cokernel.π_desc {W : C} (k : Y ⟶ W) (h : f ≫ k = 0) :
    cokernel.π f ≫ cokernel.desc f k h = k :=
  (cokernelIsCokernel f).fac (CokernelCofork.ofπ k h) WalkingParallelPair.one

-- Porting note: added to ease the port of `Abelian.Exact`

@[reassoc (attr := simp)]
lemma colimit_ι_zero_cokernel_desc {C : Type*} [Category C]
    [HasZeroMorphisms C] {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (h : f ≫ g = 0) [HasCokernel f] :
    colimit.ι (parallelPair f 0) WalkingParallelPair.zero ≫ cokernel.desc f g h = 0 := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    inst✝ : CategoryTheory.Limits.HasCokernel f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (Cat …
  -/
  rw [(colimit.w (parallelPair f 0) WalkingParallelPairHom.left).symm]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    inst✝ : CategoryTheory.Limits.HasCokernel f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem cokernel.desc_zero {W : C} {h} : cokernel.desc f (0 : Y ⟶ W) h = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasCokernel f
    W : C
    h : Eq (CategoryTheory.CategoryStruct.comp f 0) 0
    ⊢ Eq (CategoryTheory.Limits.cokernel.desc f 0 h) 0
  -/
  ext; simp
       /-
         🎉 no goals
       -/


instance cokernel.desc_epi {W : C} (k : Y ⟶ W) (h : f ≫ k = 0) [Epi k] :
    Epi (cokernel.desc f k h) :=
  ⟨fun {Z} g g' w => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCokernel f
      W : C
      k : Quiver.Hom Y W
      h : Eq (CategoryTheory.CategoryStruct.comp f k) 0
      inst✝ : CategoryTheory.Epi k
      Z : C
      g g' : Quiver.Hom W Z
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.des …
      ⊢ Eq g g'
    -/
    replace w := cokernel.π f ≫= w
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCokernel f
      W : C
      k : Quiver.Hom Y W
      h : Eq (CategoryTheory.CategoryStruct.comp f k) 0
      inst✝ : CategoryTheory.Epi k
      Z : C
      g g' : Quiver.Hom W Z
      w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π f …
      ⊢ Eq g g'
    -/
    simp only [cokernel.π_desc_assoc] at w
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCokernel f
      W : C
      k : Quiver.Hom Y W
      h : Eq (CategoryTheory.CategoryStruct.comp f k) 0
      inst✝ : CategoryTheory.Epi k
      Z : C
      g g' : Quiver.Hom W Z
      w : Eq (CategoryTheory.CategoryStruct.comp k g) (CategoryTheory.CategoryStruct …
      ⊢ Eq g g'
    -/
    exact (cancel_epi k).1 w⟩
    /-
      🎉 no goals
    -/


/-- Any morphism `k : Y ⟶ W` satisfying `f ≫ k = 0` induces `l : cokernel f ⟶ W` such that
    `cokernel.π f ≫ l = k`. -/
def cokernel.desc' {W : C} (k : Y ⟶ W) (h : f ≫ k = 0) :
    { l : cokernel f ⟶ W // cokernel.π f ≫ l = k } :=
  ⟨cokernel.desc f k h, cokernel.π_desc _ _ _⟩


/-- A commuting square induces a morphism of cokernels. -/
abbrev cokernel.map {X' Y' : C} (f' : X' ⟶ Y') [HasCokernel f'] (p : X ⟶ X') (q : Y ⟶ Y')
    (w : f ≫ q = p ≫ f') : cokernel f ⟶ cokernel f' :=
  cokernel.desc f (q ≫ cokernel.π f') (by
    have : f ≫ q ≫ π f' = p ≫ f' ≫ π f' := by
      simp only [← Category.assoc]
      apply congrArg (· ≫ π f') w
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
      X Y : C
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasCokernel f
      X' Y' : C
      f' : Quiver.Hom X' Y'
      inst✝ : CategoryTheory.Limits.HasCokernel f'
      p : Quiver.Hom X X'
      q : Quiver.Hom Y Y'
      w : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStruct …
      this : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
    -/
    simp [this])
    /-
      🎉 no goals
    -/


/-- Given a commutative diagram
    X --f--> Y --g--> Z
    |        |        |
    |        |        |
    v        v        v
    X' -f'-> Y' -g'-> Z'
with horizontal arrows composing to zero,
then we obtain a commutative square
   cokernel f ---> Z
   |               |
   | cokernel.map  |
   |               |
   v               v
   cokernel f' --> Z'
-/
theorem cokernel.map_desc {X Y Z X' Y' Z' : C} (f : X ⟶ Y) [HasCokernel f] (g : Y ⟶ Z)
    (w : f ≫ g = 0) (f' : X' ⟶ Y') [HasCokernel f'] (g' : Y' ⟶ Z') (w' : f' ≫ g' = 0) (p : X ⟶ X')
    (q : Y ⟶ Y') (r : Z ⟶ Z') (h₁ : f ≫ q = p ≫ f') (h₂ : g ≫ r = q ≫ g') :
    cokernel.map f f' p q h₁ ≫ cokernel.desc f' g' w' = cokernel.desc f g w ≫ r := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z X' Y' Z' : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    f' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasCokernel f'
    g' : Quiver.Hom Y' Z'
    w' : Eq (CategoryTheory.CategoryStruct.comp f' g') 0
    p : Quiver.Hom X X'
    q : Quiver.Hom Y Y'
    r : Quiver.Hom Z Z'
    h₁ : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStruc …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp g r) (CategoryTheory.CategoryStruc …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.map f …
  -/
  ext; simp [h₂]
       /-
         🎉 no goals
       -/


/-- A commuting square of isomorphisms induces an isomorphism of cokernels. -/
@[simps]
def cokernel.mapIso {X' Y' : C} (f' : X' ⟶ Y') [HasCokernel f'] (p : X ≅ X') (q : Y ≅ Y')
    (w : f ≫ q.hom = p.hom ≫ f') : cokernel f ≅ cokernel f' where
  hom := cokernel.map f f' p.hom q.hom w
  inv := cokernel.map f' f p.inv q.inv (by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
            X Y : C
            f : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasCokernel f
            X' Y' : C
            f' : Quiver.Hom X' Y'
            inst✝ : CategoryTheory.Limits.HasCokernel f'
            p : CategoryTheory.Iso X X'
            q : CategoryTheory.Iso Y Y'
            w : Eq (CategoryTheory.CategoryStruct.comp f q.hom) (CategoryTheory.CategorySt …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp f' q.inv) (CategoryTheory.CategoryStr …
          -/
          refine (cancel_mono q.hom).1 ?_
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
            X Y : C
            f : Quiver.Hom X Y
            inst✝¹ : CategoryTheory.Limits.HasCokernel f
            X' Y' : C
            f' : Quiver.Hom X' Y'
            inst✝ : CategoryTheory.Limits.HasCokernel f'
            p : CategoryTheory.Iso X X'
            q : CategoryTheory.Iso Y Y'
            w : Eq (CategoryTheory.CategoryStruct.comp f q.hom) (CategoryTheory.CategorySt …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
          -/
          simp [w])
          /-
            🎉 no goals
          -/


/-- The cokernel of the zero morphism is an isomorphism -/
instance cokernel.π_zero_isIso : IsIso (cokernel.π (0 : X ⟶ Y)) :=
  coequalizer.π_of_self _


theorem eq_zero_of_mono_cokernel [Mono (cokernel.π f)] : f = 0 :=
                                     /-
                                       C : Type u
                                       inst✝³ : CategoryTheory.Category.{v, u} C
                                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                       X Y : C
                                       f : Quiver.Hom X Y
                                       inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                       inst✝ : CategoryTheory.Mono (CategoryTheory.Limits.cokernel.π f)
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.cokernel.π f …
                                     -/
  (cancel_mono (cokernel.π f)).1 (by simp)
                                     /-
                                       🎉 no goals
                                     -/


/-- The cokernel of a zero morphism is isomorphic to the target. -/
def cokernelZeroIsoTarget : cokernel (0 : X ⟶ Y) ≅ Y :=
  coequalizer.isoTargetOfSelf 0


@[simp]
theorem cokernelZeroIsoTarget_hom :
                                                                    /-
                                                                      C : Type u
                                                                      inst✝² : CategoryTheory.Category.{v, u} C
                                                                      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                      X Y : C
                                                                      f : Quiver.Hom X Y
                                                                      inst✝ : CategoryTheory.Limits.HasCokernel f
                                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.CategoryStruct.id Y …
                                                                    -/
    cokernelZeroIsoTarget.hom = cokernel.desc (0 : X ⟶ Y) (𝟙 Y) (by simp) := by
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    ⊢ Eq CategoryTheory.Limits.cokernelZeroIsoTarget.hom (CategoryTheory.Limits.co …
  -/
  ext; simp [cokernelZeroIsoTarget]
       /-
         🎉 no goals
       -/


@[simp]
theorem cokernelZeroIsoTarget_inv : cokernelZeroIsoTarget.inv = cokernel.π (0 : X ⟶ Y) :=
  rfl


/-- If two morphisms are known to be equal, then their cokernels are isomorphic. -/
def cokernelIsoOfEq {f g : X ⟶ Y} [HasCokernel f] [HasCokernel g] (h : f = g) :
    cokernel f ≅ cokernel g :=
                             /-
                               C : Type u
                               inst✝⁴ : CategoryTheory.Category.{v, u} C
                               inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                               X Y : C
                               f✝ : Quiver.Hom X Y
                               inst✝² : CategoryTheory.Limits.HasCokernel f✝
                               f g : Quiver.Hom X Y
                               inst✝¹ : CategoryTheory.Limits.HasCokernel f
                               inst✝ : CategoryTheory.Limits.HasCokernel g
                               h : Eq f g
                               ⊢ CategoryTheory.Iso (CategoryTheory.Limits.parallelPair f 0) (CategoryTheory. …
                             -/
  HasColimit.isoOfNatIso (by simp [h]; rfl)
                                       /-
                                         🎉 no goals
                                       -/


@[simp]
theorem cokernelIsoOfEq_refl {h : f = f} : cokernelIsoOfEq h = Iso.refl (cokernel f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasCokernel f
    h : Eq f f
    ⊢ Eq (CategoryTheory.Limits.cokernelIsoOfEq h) (CategoryTheory.Iso.refl (Categ …
  -/
  ext; simp [cokernelIsoOfEq]
       /-
         🎉 no goals
       -/


@[reassoc (attr := simp)]
theorem π_comp_cokernelIsoOfEq_hom {f g : X ⟶ Y} [HasCokernel f] [HasCokernel g] (h : f = g) :
    cokernel.π f ≫ (cokernelIsoOfEq h).hom = cokernel.π g := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel g
    h : Eq f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π f)  …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[reassoc (attr := simp)]
theorem π_comp_cokernelIsoOfEq_inv {f g : X ⟶ Y} [HasCokernel f] [HasCokernel g] (h : f = g) :
    cokernel.π _ ≫ (cokernelIsoOfEq h).inv = cokernel.π _ := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel g
    h : Eq f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π g)  …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[reassoc (attr := simp)]
theorem cokernelIsoOfEq_hom_comp_desc {Z} {f g : X ⟶ Y} [HasCokernel f] [HasCokernel g] (h : f = g)
    (e : Y ⟶ Z) (he) :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                             X Y : C
                                                                             f✝ : Quiver.Hom X Y
                                                                             inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                                                             Z : C
                                                                             f g : Quiver.Hom X Y
                                                                             inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                                                             inst✝ : CategoryTheory.Limits.HasCokernel g
                                                                             h : Eq f g
                                                                             e : Quiver.Hom Y Z
                                                                             he : Eq (CategoryTheory.CategoryStruct.comp g e) 0
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f e) 0
                                                                           -/
    (cokernelIsoOfEq h).hom ≫ cokernel.desc _ e he = cokernel.desc _ e (by simp [h, he]) := by
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel g
    h : Eq f g
    e : Quiver.Hom Y Z
    he : Eq (CategoryTheory.CategoryStruct.comp g e) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernelIsoOfE …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[reassoc (attr := simp)]
theorem cokernelIsoOfEq_inv_comp_desc {Z} {f g : X ⟶ Y} [HasCokernel f] [HasCokernel g] (h : f = g)
    (e : Y ⟶ Z) (he) :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                             inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                             X Y : C
                                                                             f✝ : Quiver.Hom X Y
                                                                             inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                                                             Z : C
                                                                             f g : Quiver.Hom X Y
                                                                             inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                                                             inst✝ : CategoryTheory.Limits.HasCokernel g
                                                                             h : Eq f g
                                                                             e : Quiver.Hom Y Z
                                                                             he : Eq (CategoryTheory.CategoryStruct.comp f e) 0
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp g e) 0
                                                                           -/
    (cokernelIsoOfEq h).inv ≫ cokernel.desc _ e he = cokernel.desc _ e (by simp [← h, he]) := by
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X Y Z : C
    f g : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel g
    h : Eq f g
    e : Quiver.Hom Y Z
    he : Eq (CategoryTheory.CategoryStruct.comp f e) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernelIsoOfE …
  -/
  cases h; simp
           /-
             🎉 no goals
           -/


@[simp]
theorem cokernelIsoOfEq_trans {f g h : X ⟶ Y} [HasCokernel f] [HasCokernel g] [HasCokernel h]
    (w₁ : f = g) (w₂ : g = h) :
    cokernelIsoOfEq w₁ ≪≫ cokernelIsoOfEq w₂ = cokernelIsoOfEq (w₁.trans w₂) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f g h : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasCokernel f
    inst✝¹ : CategoryTheory.Limits.HasCokernel g
    inst✝ : CategoryTheory.Limits.HasCokernel h
    w₁ : Eq f g
    w₂ : Eq g h
    ⊢ Eq ((CategoryTheory.Limits.cokernelIsoOfEq w₁).trans (CategoryTheory.Limits. …
  -/
  cases w₁; cases w₂; ext; simp [cokernelIsoOfEq]
                           /-
                             🎉 no goals
                           -/


theorem cokernel_not_mono_of_nonzero (w : f ≠ 0) : ¬Mono (cokernel.π f) := fun _ =>
  w (eq_zero_of_mono_cokernel f)


theorem cokernel_not_iso_of_nonzero (w : f ≠ 0) : IsIso (cokernel.π f) → False := fun _ =>
  cokernel_not_mono_of_nonzero w inferInstance


instance hasCokernel_comp_iso {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasCokernel f] [IsIso g] :
    HasCokernel (f ≫ g) where
  exists_colimit :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                  X✝ Y✝ : C
                                                                  f✝ : Quiver.Hom X✝ Y✝
                                                                  inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                                                  X Y Z : C
                                                                  f : Quiver.Hom X Y
                                                                  g : Quiver.Hom Y Z
                                                                  inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                                                  inst✝ : CategoryTheory.IsIso g
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                                                -/
    ⟨{  cocone := CokernelCofork.ofπ (inv g ≫ cokernel.π f) (by simp)
                                                                /-
                                                                  🎉 no goals
                                                                -/
        isColimit :=
          isColimitAux _
            (fun s =>
                                            /-
                                              C : Type u
                                              inst✝⁴ : CategoryTheory.Category.{v, u} C
                                              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                              X✝ Y✝ : C
                                              f✝ : Quiver.Hom X✝ Y✝
                                              inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                              X Y Z : C
                                              f : Quiver.Hom X Y
                                              g : Quiver.Hom Y Z
                                              inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                              inst✝ : CategoryTheory.IsIso g
                                              s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.CategoryStruct.comp f …
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                            -/
              cokernel.desc _ (g ≫ s.π) (by rw [← Category.assoc, CokernelCofork.condition]))
                                            /-
                                              🎉 no goals
                                            -/
                /-
                  C : Type u
                  inst✝⁴ : CategoryTheory.Category.{v, u} C
                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                  X✝ Y✝ : C
                  f✝ : Quiver.Hom X✝ Y✝
                  inst✝² : CategoryTheory.Limits.HasCokernel f✝
                  X Y Z : C
                  f : Quiver.Hom X Y
                  g : Quiver.Hom Y Z
                  inst✝¹ : CategoryTheory.Limits.HasCokernel f
                  inst✝ : CategoryTheory.IsIso g
                  ⊢ ∀ (s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.CategoryStruct.c …
                -/
            (by aesop_cat) fun s m w => by
                /-
                  🎉 no goals
                -/
            /-
              C : Type u
              inst✝⁴ : CategoryTheory.Category.{v, u} C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              X✝ Y✝ : C
              f✝ : Quiver.Hom X✝ Y✝
              inst✝² : CategoryTheory.Limits.HasCokernel f✝
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasCokernel f
              inst✝ : CategoryTheory.IsIso g
              s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.CategoryStruct.comp f …
              m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Categ …
              w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
              ⊢ Eq m ((fun s => CategoryTheory.Limits.cokernel.desc f (CategoryTheory.Catego …
            -/
            simp_rw [← w]
            /-
              C : Type u
              inst✝⁴ : CategoryTheory.Category.{v, u} C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              X✝ Y✝ : C
              f✝ : Quiver.Hom X✝ Y✝
              inst✝² : CategoryTheory.Limits.HasCokernel f✝
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasCokernel f
              inst✝ : CategoryTheory.IsIso g
              s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.CategoryStruct.comp f …
              m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Categ …
              w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
              ⊢ Eq m (CategoryTheory.Limits.cokernel.desc f (CategoryTheory.CategoryStruct.c …
            -/
            symm
            /-
              C : Type u
              inst✝⁴ : CategoryTheory.Category.{v, u} C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              X✝ Y✝ : C
              f✝ : Quiver.Hom X✝ Y✝
              inst✝² : CategoryTheory.Limits.HasCokernel f✝
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasCokernel f
              inst✝ : CategoryTheory.IsIso g
              s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.CategoryStruct.comp f …
              m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Categ …
              w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
              ⊢ Eq (CategoryTheory.Limits.cokernel.desc f (CategoryTheory.CategoryStruct.com …
            -/
            apply coequalizer.hom_ext
            /-
              case h
              C : Type u
              inst✝⁴ : CategoryTheory.Category.{v, u} C
              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
              X✝ Y✝ : C
              f✝ : Quiver.Hom X✝ Y✝
              inst✝² : CategoryTheory.Limits.HasCokernel f✝
              X Y Z : C
              f : Quiver.Hom X Y
              g : Quiver.Hom Y Z
              inst✝¹ : CategoryTheory.Limits.HasCokernel f
              inst✝ : CategoryTheory.IsIso g
              s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.CategoryStruct.comp f …
              m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Categ …
              w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
            -/
            simp }⟩
            /-
              🎉 no goals
            -/


/-- When `g` is an isomorphism, the cokernel of `f ≫ g` is isomorphic to the cokernel of `f`.
-/
@[simps]
def cokernelCompIsIso {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [HasCokernel f] [IsIso g] :
    cokernel (f ≫ g) ≅ cokernel f where
                                                    /-
                                                      C : Type u
                                                      inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                      X✝ Y✝ : C
                                                      f✝ : Quiver.Hom X✝ Y✝
                                                      inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                                      X Y Z : C
                                                      f : Quiver.Hom X Y
                                                      g : Quiver.Hom Y Z
                                                      inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                                      inst✝ : CategoryTheory.IsIso g
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                                    -/
  hom := cokernel.desc _ (inv g ≫ cokernel.π f) (by simp)
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                      /-
                                                        C : Type u
                                                        inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                                        X✝ Y✝ : C
                                                        f✝ : Quiver.Hom X✝ Y✝
                                                        inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                                        X Y Z : C
                                                        f : Quiver.Hom X Y
                                                        g : Quiver.Hom Y Z
                                                        inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                                        inst✝ : CategoryTheory.IsIso g
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                                      -/
  inv := cokernel.desc _ (g ≫ cokernel.π (f ≫ g)) (by rw [← Category.assoc, cokernel.condition])
                                                      /-
                                                        🎉 no goals
                                                      -/


instance hasCokernel_epi_comp {X Y : C} (f : X ⟶ Y) [HasCokernel f] {W} (g : W ⟶ X) [Epi g] :
    HasCokernel (g ≫ f) :=
  ⟨⟨{   cocone := _
        isColimit := isCokernelEpiComp (colimit.isColimit _) g rfl }⟩⟩


/-- When `f` is an epimorphism, the cokernel of `f ≫ g` is isomorphic to the cokernel of `g`.
-/
@[simps]
def cokernelEpiComp {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) [Epi f] [HasCokernel g] :
    cokernel (f ≫ g) ≅ cokernel g where
                                            /-
                                              C : Type u
                                              inst✝⁴ : CategoryTheory.Category.{v, u} C
                                              inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                              X✝ Y✝ : C
                                              f✝ : Quiver.Hom X✝ Y✝
                                              inst✝² : CategoryTheory.Limits.HasCokernel f✝
                                              X Y Z : C
                                              f : Quiver.Hom X Y
                                              g : Quiver.Hom Y Z
                                              inst✝¹ : CategoryTheory.Epi f
                                              inst✝ : CategoryTheory.Limits.HasCokernel g
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                            -/
  hom := cokernel.desc _ (cokernel.π g) (by simp)
                                            /-
                                              🎉 no goals
                                            -/
  inv :=
    cokernel.desc _ (cokernel.π (f ≫ g))
      (by
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          inst✝² : CategoryTheory.Limits.HasCokernel f✝
          X Y Z : C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          inst✝¹ : CategoryTheory.Epi f
          inst✝ : CategoryTheory.Limits.HasCokernel g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.cokernel.π ( …
        -/
        rw [← cancel_epi f, ← Category.assoc]
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          inst✝² : CategoryTheory.Limits.HasCokernel f✝
          X Y Z : C
          f : Quiver.Hom X Y
          g : Quiver.Hom Y Z
          inst✝¹ : CategoryTheory.Epi f
          inst✝ : CategoryTheory.Limits.HasCokernel g
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
        -/
        simp)
        /-
          🎉 no goals
        -/


/-- The morphism to the zero object determines a cocone on a cokernel diagram -/
def cokernel.zeroCokernelCofork : CokernelCofork f where
  pt := 0
  ι := { app := fun _ => 0 }


/-- The morphism to the zero object is a cokernel of an epimorphism -/
def cokernel.isColimitCoconeZeroCocone [Epi f] : IsColimit (cokernel.zeroCokernelCofork f) :=
  Cofork.IsColimit.mk _ (fun _ => 0)
    (fun s => by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Epi f
        s : CategoryTheory.Limits.Cofork f 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
      -/
      erw [zero_comp]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Epi f
        s : CategoryTheory.Limits.Cofork f 0
        ⊢ Eq 0 s.π
      -/
      refine (zero_of_epi_comp f ?_).symm
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        inst✝¹ : CategoryTheory.Limits.HasZeroObject C
        inst✝ : CategoryTheory.Epi f
        s : CategoryTheory.Limits.Cofork f 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.π) 0
      -/
      exact CokernelCofork.condition _)
      /-
        🎉 no goals
      -/
    fun _ _ _ => zero_of_from_zero _


/-- The cokernel of an epimorphism is isomorphic to the zero object -/
def cokernel.ofEpi [HasCokernel f] [Epi f] : cokernel f ≅ 0 :=
  Functor.mapIso (Cocones.forget _) <|
    IsColimit.uniqueUpToIso (colimit.isColimit (parallelPair f 0))
      (cokernel.isColimitCoconeZeroCocone f)


/-- The cokernel morphism of an epimorphism is a zero morphism -/
theorem cokernel.π_of_epi [HasCokernel f] [Epi f] : cokernel.π f = 0 :=
  zero_of_target_iso_zero _ (cokernel.ofEpi f)


@[simp]
theorem MonoFactorisation.kernel_ι_comp [HasKernel f] (F : MonoFactorisation f) :
    kernel.ι f ≫ F.e = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    F : CategoryTheory.Limits.MonoFactorisation f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) F. …
  -/
  rw [← cancel_mono F.m, zero_comp, Category.assoc, F.fac, kernel.condition]
  /-
    🎉 no goals
  -/


/-- The cokernel of the image inclusion of a morphism `f` is isomorphic to the cokernel of `f`.

(This result requires that the factorisation through the image is an epimorphism.
This holds in any category with equalizers.)
-/
@[simps]
def cokernelImageι {X Y : C} (f : X ⟶ Y) [HasImage f] [HasCokernel (image.ι f)] [HasCokernel f]
    [Epi (factorThruImage f)] : cokernel (image.ι f) ≅ cokernel f where
  hom :=
    cokernel.desc _ (cokernel.π f)
      (by
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          X Y : C
          f : Quiver.Hom X Y
          inst✝³ : CategoryTheory.Limits.HasImage f
          inst✝² : CategoryTheory.Limits.HasCokernel (CategoryTheory.Limits.image.ι f)
          inst✝¹ : CategoryTheory.Limits.HasCokernel f
          inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImage f)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.ι f) (Ca …
        -/
        have w := cokernel.condition f
        conv at w =>
          lhs
          congr
          rw [← image.fac f]
        rw [← HasZeroMorphisms.comp_zero (Limits.factorThruImage f), Category.assoc,
          cancel_epi] at w
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          X Y : C
          f : Quiver.Hom X Y
          inst✝³ : CategoryTheory.Limits.HasImage f
          inst✝² : CategoryTheory.Limits.HasCokernel (CategoryTheory.Limits.image.ι f)
          inst✝¹ : CategoryTheory.Limits.HasCokernel f
          inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImage f)
          w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.ι f) ( …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.ι f) (Ca …
        -/
        exact w)
        /-
          🎉 no goals
        -/
  inv :=
    cokernel.desc _ (cokernel.π _)
      (by
        conv =>
          lhs
          congr
          rw [← image.fac f]
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
          X✝ Y✝ : C
          f✝ : Quiver.Hom X✝ Y✝
          X Y : C
          f : Quiver.Hom X Y
          inst✝³ : CategoryTheory.Limits.HasImage f
          inst✝² : CategoryTheory.Limits.HasCokernel (CategoryTheory.Limits.image.ι f)
          inst✝¹ : CategoryTheory.Limits.HasCokernel f
          inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImage f)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [Category.assoc, cokernel.condition, HasZeroMorphisms.comp_zero])
        /-
          🎉 no goals
        -/


/-- The cokernel of a zero morphism is an isomorphism -/
theorem cokernel.π_of_zero : IsIso (cokernel.π (0 : X ⟶ Y)) :=
  coequalizer.π_of_self _


/-- The kernel of the cokernel of an epimorphism is an isomorphism -/
instance kernel.of_cokernel_of_epi [HasCokernel f] [HasKernel (cokernel.π f)] [Epi f] :
    IsIso (kernel.ι (cokernel.π f)) :=
  equalizer.ι_of_eq <| cokernel.π_of_epi f


/-- The cokernel of the kernel of a monomorphism is an isomorphism -/
instance cokernel.of_kernel_of_mono [HasKernel f] [HasCokernel (kernel.ι f)] [Mono f] :
    IsIso (cokernel.π (kernel.ι f)) :=
  coequalizer.π_of_eq <| kernel.ι_of_mono f


/-- If `f ≫ g = 0` implies `g = 0` for all `g`, then `0 : Y ⟶ 0` is a cokernel of `f`. -/
def zeroCokernelOfZeroCancel {X Y : C} (f : X ⟶ Y)
    (hf : ∀ (Z : C) (g : Y ⟶ Z) (_ : f ≫ g = 0), g = 0) :
                                                                 /-
                                                                   C : Type u
                                                                   inst✝² : CategoryTheory.Category.{v, u} C
                                                                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                   X✝ Y✝ : C
                                                                   f✝ : Quiver.Hom X✝ Y✝
                                                                   inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                                   X Y : C
                                                                   f : Quiver.Hom X Y
                                                                   hf : ∀ (Z : C) (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp f  …
                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp f 0) 0
                                                                 -/
    IsColimit (CokernelCofork.ofπ (0 : Y ⟶ 0) (show f ≫ 0 = 0 by simp)) :=
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  Cofork.IsColimit.mk _ (fun _ => 0)
                 /-
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                   X✝ Y✝ : C
                   f✝ : Quiver.Hom X✝ Y✝
                   inst✝ : CategoryTheory.Limits.HasZeroObject C
                   X Y : C
                   f : Quiver.Hom X Y
                   hf : ∀ (Z : C) (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp f  …
                   s : CategoryTheory.Limits.Cofork f 0
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                 -/
    (fun s => by rw [hf _ _ (CokernelCofork.condition s), comp_zero]) fun s m _ => by
                 /-
                   🎉 no goals
                 -/
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
        X✝ Y✝ : C
        f✝ : Quiver.Hom X✝ Y✝
        inst✝ : CategoryTheory.Limits.HasZeroObject C
        X Y : C
        f : Quiver.Hom X Y
        hf : ∀ (Z : C) (g : Quiver.Hom Y Z), Eq (CategoryTheory.CategoryStruct.comp f  …
        s : CategoryTheory.Limits.Cofork f 0
        m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ 0 ⋯).pt s.pt
        x✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
        ⊢ Eq m ((fun x => 0) s)
      -/
      apply HasZeroObject.from_zero_ext
      /-
        🎉 no goals
      -/


/-- If `i` is an isomorphism such that `i.hom ≫ l = f`, then any cokernel of `f` is a cokernel of
    `l`. -/
def IsCokernel.ofIsoComp {Z : C} (l : Z ⟶ Y) (i : X ≅ Z) (h : i.hom ≫ l = f) {s : CokernelCofork f}
    (hs : IsColimit s) :
    IsColimit
                                                                     /-
                                                                       C : Type u
                                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                       X Y : C
                                                                       f : Quiver.Hom X Y
                                                                       Z : C
                                                                       l : Quiver.Hom Z Y
                                                                       i : CategoryTheory.Iso X Z
                                                                       h : Eq (CategoryTheory.CategoryStruct.comp i.hom l) f
                                                                       s : CategoryTheory.Limits.CokernelCofork f
                                                                       hs : CategoryTheory.Limits.IsColimit s
                                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.Cofork.π s)) 0
                                                                     -/
      (CokernelCofork.ofπ (Cofork.π s) <| show l ≫ Cofork.π s = 0 by simp [i.eq_inv_comp.2 h]) :=
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
                                                                                   /-
                                                                                     C : Type u
                                                                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                     X Y : C
                                                                                     f : Quiver.Hom X Y
                                                                                     Z : C
                                                                                     l : Quiver.Hom Z Y
                                                                                     i : CategoryTheory.Iso X Z
                                                                                     h : Eq (CategoryTheory.CategoryStruct.comp i.hom l) f
                                                                                     s✝ : CategoryTheory.Limits.CokernelCofork f
                                                                                     hs : CategoryTheory.Limits.IsColimit s✝
                                                                                     s : CategoryTheory.Limits.Cofork l 0
                                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f s.π) 0
                                                                                   -/
  Cofork.IsColimit.mk _ (fun s => hs.desc <| CokernelCofork.ofπ (Cofork.π s) <| by simp [← h])
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                   X Y : C
                   f : Quiver.Hom X Y
                   Z : C
                   l : Quiver.Hom Z Y
                   i : CategoryTheory.Iso X Z
                   h : Eq (CategoryTheory.CategoryStruct.comp i.hom l) f
                   s✝ : CategoryTheory.Limits.CokernelCofork f
                   hs : CategoryTheory.Limits.IsColimit s✝
                   s : CategoryTheory.Limits.Cofork l 0
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                 -/
    (fun s => by simp) fun s m h => by
                 /-
                   🎉 no goals
                 -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        l : Quiver.Hom Z Y
        i : CategoryTheory.Iso X Z
        h✝ : Eq (CategoryTheory.CategoryStruct.comp i.hom l) f
        s✝ : CategoryTheory.Limits.CokernelCofork f
        hs : CategoryTheory.Limits.IsColimit s✝
        s : CategoryTheory.Limits.Cofork l 0
        m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limit …
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
        ⊢ Eq m ((fun s => hs.desc (CategoryTheory.Limits.CokernelCofork.ofπ s.π ⋯)) s)
      -/
      apply Cofork.IsColimit.hom_ext hs
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        l : Quiver.Hom Z Y
        i : CategoryTheory.Iso X Z
        h✝ : Eq (CategoryTheory.CategoryStruct.comp i.hom l) f
        s✝ : CategoryTheory.Limits.CokernelCofork f
        hs : CategoryTheory.Limits.IsColimit s✝
        s : CategoryTheory.Limits.Cofork l 0
        m : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limit …
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π s✝) m …
      -/
      simpa using h
      /-
        🎉 no goals
      -/


/-- If `i` is an isomorphism such that `i.hom ≫ l = f`, then the cokernel of `f` is a cokernel of
    `l`. -/
def cokernel.ofIsoComp [HasCokernel f] {Z : C} (l : Z ⟶ Y) (i : X ≅ Z) (h : i.hom ≫ l = f) :
    IsColimit
      (CokernelCofork.ofπ (cokernel.π f) <|
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                       X Y : C
                                       f : Quiver.Hom X Y
                                       inst✝ : CategoryTheory.Limits.HasCokernel f
                                       Z : C
                                       l : Quiver.Hom Z Y
                                       i : CategoryTheory.Iso X Z
                                       h : Eq (CategoryTheory.CategoryStruct.comp i.hom l) f
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheory.Limits.cokernel.π f …
                                     -/
        show l ≫ cokernel.π f = 0 by simp [i.eq_inv_comp.2 h]) :=
                                     /-
                                       🎉 no goals
                                     -/
  IsCokernel.ofIsoComp f l i h <| colimit.isColimit _


/-- If `s` is any colimit cokernel cocone over `f` and `i` is an isomorphism such that
    `s.π ≫ i.hom = l`, then `l` is a cokernel of `f`. -/
def IsCokernel.cokernelIso {Z : C} (l : Y ⟶ Z) {s : CokernelCofork f} (hs : IsColimit s)
    (i : s.pt ≅ Z) (h : Cofork.π s ≫ i.hom = l) :
                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                           X Y : C
                                                           f : Quiver.Hom X Y
                                                           Z : C
                                                           l : Quiver.Hom Y Z
                                                           s : CategoryTheory.Limits.CokernelCofork f
                                                           hs : CategoryTheory.Limits.IsColimit s
                                                           i : CategoryTheory.Iso s.pt Z
                                                           h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π s)  …
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f l) 0
                                                         -/
    IsColimit (CokernelCofork.ofπ l <| show f ≫ l = 0 by simp [← h]) :=
                                                         /-
                                                           🎉 no goals
                                                         -/
  IsColimit.ofIsoColimit hs <|
    Cocones.ext i fun j => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        X Y : C
        f : Quiver.Hom X Y
        Z : C
        l : Quiver.Hom Y Z
        s : CategoryTheory.Limits.CokernelCofork f
        hs : CategoryTheory.Limits.IsColimit s
        i : CategoryTheory.Iso s.pt Z
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π s)  …
        j : CategoryTheory.Limits.WalkingParallelPair
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app j) i.hom) ((CategoryTheory.L …
      -/
      cases j
        /-
          case zero
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          Z : C
          l : Quiver.Hom Y Z
          s : CategoryTheory.Limits.CokernelCofork f
          hs : CategoryTheory.Limits.IsColimit s
          i : CategoryTheory.Iso s.pt Z
          h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π s)  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
        -/
      · dsimp; rw [← h]; simp
                         /-
                           🎉 no goals
                         -/
        /-
          case one
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          Z : C
          l : Quiver.Hom Y Z
          s : CategoryTheory.Limits.CokernelCofork f
          hs : CategoryTheory.Limits.IsColimit s
          i : CategoryTheory.Iso s.pt Z
          h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π s)  …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.ι.app CategoryTheory.Limits.Walkin …
        -/
      · exact h
        /-
          🎉 no goals
        -/


/-- If `i` is an isomorphism such that `cokernel.π f ≫ i.hom = l`, then `l` is a cokernel of `f`. -/
def cokernel.cokernelIso [HasCokernel f] {Z : C} (l : Y ⟶ Z) (i : cokernel f ≅ Z)
    (h : cokernel.π f ≫ i.hom = l) :
                                                         /-
                                                           C : Type u
                                                           inst✝² : CategoryTheory.Category.{v, u} C
                                                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                           X Y : C
                                                           f : Quiver.Hom X Y
                                                           inst✝ : CategoryTheory.Limits.HasCokernel f
                                                           Z : C
                                                           l : Quiver.Hom Y Z
                                                           i : CategoryTheory.Iso (CategoryTheory.Limits.cokernel f) Z
                                                           h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π f …
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f l) 0
                                                         -/
    IsColimit (@CokernelCofork.ofπ _ _ _ _ _ f _ l <| by simp [← h]) :=
                                                         /-
                                                           🎉 no goals
                                                         -/
  IsCokernel.cokernelIso f l (colimit.isColimit _) i h


/-- The comparison morphism for the kernel of `f`.
This is an isomorphism iff `G` preserves the kernel of `f`; see
`CategoryTheory/Limits/Preserves/Shapes/Kernels.lean`
-/
def kernelComparison [HasKernel f] [HasKernel (G.map f)] : G.obj (kernel f) ⟶ kernel (G.map f) :=
  kernel.lift _ (G.map (kernel.ι f))
        /-
          C : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} C
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          D : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
          G : CategoryTheory.Functor C D
          inst✝² : G.PreservesZeroMorphisms
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Limits.HasKernel (G.map f)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.kernel. …
        -/
    (by simp only [← G.map_comp, kernel.condition, Functor.map_zero])
        /-
          🎉 no goals
        -/


@[reassoc (attr := simp)]
theorem kernelComparison_comp_ι [HasKernel f] [HasKernel (G.map f)] :
    kernelComparison f G ≫ kernel.ι (G.map f) = G.map (kernel.ι f) :=
  kernel.lift_ι _ _ _


@[reassoc (attr := simp)]
theorem map_lift_kernelComparison [HasKernel f] [HasKernel (G.map f)] {Z : C} {h : Z ⟶ X}
    (w : h ≫ f = 0) :
    G.map (kernel.lift _ h w) ≫ kernelComparison f G =
                                  /-
                                    C : Type u
                                    inst✝⁶ : CategoryTheory.Category.{v, u} C
                                    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                    X Y : C
                                    f : Quiver.Hom X Y
                                    D : Type u₂
                                    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                                    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                                    G : CategoryTheory.Functor C D
                                    inst✝² : G.PreservesZeroMorphisms
                                    inst✝¹ : CategoryTheory.Limits.HasKernel f
                                    inst✝ : CategoryTheory.Limits.HasKernel (G.map f)
                                    Z : C
                                    h : Quiver.Hom Z X
                                    w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) 0
                                  -/
      kernel.lift _ (G.map h) (by simp only [← G.map_comp, w, Functor.map_zero]) := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel (G.map f)
    Z : C
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.kernel. …
  -/
  ext; simp [← G.map_comp]
       /-
         🎉 no goals
       -/


@[reassoc]
theorem kernelComparison_comp_kernel_map {X' Y' : C} [HasKernel f] [HasKernel (G.map f)]
    (g : X' ⟶ Y') [HasKernel g] [HasKernel (G.map g)] (p : X ⟶ X') (q : Y ⟶ Y')
    (hpq : f ≫ q = p ≫ g) :
    kernelComparison f G ≫
                                                               /-
                                                                 C : Type u
                                                                 inst✝⁸ : CategoryTheory.Category.{v, u} C
                                                                 inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                 X Y : C
                                                                 f : Quiver.Hom X Y
                                                                 D : Type u₂
                                                                 inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                                                                 inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
                                                                 G : CategoryTheory.Functor C D
                                                                 inst✝⁴ : G.PreservesZeroMorphisms
                                                                 X' Y' : C
                                                                 inst✝³ : CategoryTheory.Limits.HasKernel f
                                                                 inst✝² : CategoryTheory.Limits.HasKernel (G.map f)
                                                                 g : Quiver.Hom X' Y'
                                                                 inst✝¹ : CategoryTheory.Limits.HasKernel g
                                                                 inst✝ : CategoryTheory.Limits.HasKernel (G.map g)
                                                                 p : Quiver.Hom X X'
                                                                 q : Quiver.Hom Y Y'
                                                                 hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map q)) (CategoryTheory. …
                                                               -/
        kernel.map (G.map f) (G.map g) (G.map p) (G.map q) (by rw [← G.map_comp, hpq, G.map_comp]) =
                                                               /-
                                                                 🎉 no goals
                                                               -/
      G.map (kernel.map f g p q hpq) ≫ kernelComparison g G :=
                          /-
                            C : Type u
                            inst✝⁸ : CategoryTheory.Category.{v, u} C
                            inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
                            X Y : C
                            f : Quiver.Hom X Y
                            D : Type u₂
                            inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                            inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
                            G : CategoryTheory.Functor C D
                            inst✝⁴ : G.PreservesZeroMorphisms
                            X' Y' : C
                            inst✝³ : CategoryTheory.Limits.HasKernel f
                            inst✝² : CategoryTheory.Limits.HasKernel (G.map f)
                            g : Quiver.Hom X' Y'
                            inst✝¹ : CategoryTheory.Limits.HasKernel g
                            inst✝ : CategoryTheory.Limits.HasKernel (G.map g)
                            p : Quiver.Hom X X'
                            q : Quiver.Hom Y Y'
                            hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.kernel. …
                          -/
  kernel.lift_map _ _ (by rw [← G.map_comp, kernel.condition, G.map_zero]) _ _
                          /-
                            🎉 no goals
                          -/
        /-
          C : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} C
          inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          D : Type u₂
          inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
          G : CategoryTheory.Functor C D
          inst✝⁴ : G.PreservesZeroMorphisms
          X' Y' : C
          inst✝³ : CategoryTheory.Limits.HasKernel f
          inst✝² : CategoryTheory.Limits.HasKernel (G.map f)
          g : Quiver.Hom X' Y'
          inst✝¹ : CategoryTheory.Limits.HasKernel g
          inst✝ : CategoryTheory.Limits.HasKernel (G.map g)
          p : Quiver.Hom X X'
          q : Quiver.Hom Y Y'
          hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.kernel. …
        -/
    (by rw [← G.map_comp, kernel.condition, G.map_zero]) _ _ _
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} C
          inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          D : Type u₂
          inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
          G : CategoryTheory.Functor C D
          inst✝⁴ : G.PreservesZeroMorphisms
          X' Y' : C
          inst✝³ : CategoryTheory.Limits.HasKernel f
          inst✝² : CategoryTheory.Limits.HasKernel (G.map f)
          g : Quiver.Hom X' Y'
          inst✝¹ : CategoryTheory.Limits.HasKernel g
          inst✝ : CategoryTheory.Limits.HasKernel (G.map g)
          p : Quiver.Hom X X'
          q : Quiver.Hom Y Y'
          hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.kernel. …
        -/
    (by simp only [← G.map_comp]; exact G.congr_map (kernel.lift_ι _ _ _).symm) _
                                  /-
                                    🎉 no goals
                                  -/


/-- The comparison morphism for the cokernel of `f`. -/
def cokernelComparison [HasCokernel f] [HasCokernel (G.map f)] :
    cokernel (G.map f) ⟶ G.obj (cokernel f) :=
  cokernel.desc _ (G.map (coequalizer.π _ _))
        /-
          C : Type u
          inst✝⁶ : CategoryTheory.Category.{v, u} C
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          D : Type u₂
          inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
          inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
          G : CategoryTheory.Functor C D
          inst✝² : G.PreservesZeroMorphisms
          inst✝¹ : CategoryTheory.Limits.HasCokernel f
          inst✝ : CategoryTheory.Limits.HasCokernel (G.map f)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
        -/
    (by simp only [← G.map_comp, cokernel.condition, Functor.map_zero])
        /-
          🎉 no goals
        -/


@[reassoc (attr := simp)]
theorem π_comp_cokernelComparison [HasCokernel f] [HasCokernel (G.map f)] :
    cokernel.π (G.map f) ≫ cokernelComparison f G = G.map (cokernel.π _) :=
  cokernel.π_desc _ _ _


@[reassoc (attr := simp)]
theorem cokernelComparison_map_desc [HasCokernel f] [HasCokernel (G.map f)] {Z : C} {h : Y ⟶ Z}
    (w : f ≫ h = 0) :
    cokernelComparison f G ≫ G.map (cokernel.desc _ h w) =
                                    /-
                                      C : Type u
                                      inst✝⁶ : CategoryTheory.Category.{v, u} C
                                      inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                                      X Y : C
                                      f : Quiver.Hom X Y
                                      D : Type u₂
                                      inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                                      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                                      G : CategoryTheory.Functor C D
                                      inst✝² : G.PreservesZeroMorphisms
                                      inst✝¹ : CategoryTheory.Limits.HasCokernel f
                                      inst✝ : CategoryTheory.Limits.HasCokernel (G.map f)
                                      Z : C
                                      h : Quiver.Hom Y Z
                                      w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) 0
                                    -/
      cokernel.desc _ (G.map h) (by simp only [← G.map_comp, w, Functor.map_zero]) := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    X Y : C
    f : Quiver.Hom X Y
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel (G.map f)
    Z : C
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernelCompar …
  -/
  ext; simp [← G.map_comp]
       /-
         🎉 no goals
       -/


@[reassoc]
theorem cokernel_map_comp_cokernelComparison {X' Y' : C} [HasCokernel f] [HasCokernel (G.map f)]
    (g : X' ⟶ Y') [HasCokernel g] [HasCokernel (G.map g)] (p : X ⟶ X') (q : Y ⟶ Y')
    (hpq : f ≫ q = p ≫ g) :
                                                             /-
                                                               C : Type u
                                                               inst✝⁸ : CategoryTheory.Category.{v, u} C
                                                               inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
                                                               X Y : C
                                                               f : Quiver.Hom X Y
                                                               D : Type u₂
                                                               inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                                                               inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
                                                               G : CategoryTheory.Functor C D
                                                               inst✝⁴ : G.PreservesZeroMorphisms
                                                               X' Y' : C
                                                               inst✝³ : CategoryTheory.Limits.HasCokernel f
                                                               inst✝² : CategoryTheory.Limits.HasCokernel (G.map f)
                                                               g : Quiver.Hom X' Y'
                                                               inst✝¹ : CategoryTheory.Limits.HasCokernel g
                                                               inst✝ : CategoryTheory.Limits.HasCokernel (G.map g)
                                                               p : Quiver.Hom X X'
                                                               q : Quiver.Hom Y Y'
                                                               hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
                                                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map q)) (CategoryTheory. …
                                                             -/
    cokernel.map (G.map f) (G.map g) (G.map p) (G.map q) (by rw [← G.map_comp, hpq, G.map_comp]) ≫
                                                             /-
                                                               🎉 no goals
                                                             -/
        cokernelComparison _ G =
      cokernelComparison _ G ≫ G.map (cokernel.map f g p q hpq) :=
                            /-
                              C : Type u
                              inst✝⁸ : CategoryTheory.Category.{v, u} C
                              inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
                              X Y : C
                              f : Quiver.Hom X Y
                              D : Type u₂
                              inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
                              inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
                              G : CategoryTheory.Functor C D
                              inst✝⁴ : G.PreservesZeroMorphisms
                              X' Y' : C
                              inst✝³ : CategoryTheory.Limits.HasCokernel f
                              inst✝² : CategoryTheory.Limits.HasCokernel (G.map f)
                              g : Quiver.Hom X' Y'
                              inst✝¹ : CategoryTheory.Limits.HasCokernel g
                              inst✝ : CategoryTheory.Limits.HasCokernel (G.map g)
                              p : Quiver.Hom X X'
                              q : Quiver.Hom Y Y'
                              hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
                            -/
  cokernel.map_desc _ _ (by rw [← G.map_comp, cokernel.condition, G.map_zero]) _ _
                            /-
                              🎉 no goals
                            -/
        /-
          C : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} C
          inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          D : Type u₂
          inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
          G : CategoryTheory.Functor C D
          inst✝⁴ : G.PreservesZeroMorphisms
          X' Y' : C
          inst✝³ : CategoryTheory.Limits.HasCokernel f
          inst✝² : CategoryTheory.Limits.HasCokernel (G.map f)
          g : Quiver.Hom X' Y'
          inst✝¹ : CategoryTheory.Limits.HasCokernel g
          inst✝ : CategoryTheory.Limits.HasCokernel (G.map g)
          p : Quiver.Hom X X'
          q : Quiver.Hom Y Y'
          hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map g) (G.map (CategoryTheory.Limi …
        -/
    (by rw [← G.map_comp, cokernel.condition, G.map_zero]) _ _ _ _
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝⁸ : CategoryTheory.Category.{v, u} C
          inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms C
          X Y : C
          f : Quiver.Hom X Y
          D : Type u₂
          inst✝⁶ : CategoryTheory.Category.{v₂, u₂} D
          inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms D
          G : CategoryTheory.Functor C D
          inst✝⁴ : G.PreservesZeroMorphisms
          X' Y' : C
          inst✝³ : CategoryTheory.Limits.HasCokernel f
          inst✝² : CategoryTheory.Limits.HasCokernel (G.map f)
          g : Quiver.Hom X' Y'
          inst✝¹ : CategoryTheory.Limits.HasCokernel g
          inst✝ : CategoryTheory.Limits.HasCokernel (G.map g)
          p : Quiver.Hom X X'
          q : Quiver.Hom Y Y'
          hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.coequal …
        -/
    (by simp only [← G.map_comp]; exact G.congr_map (cokernel.π_desc _ _ _))
                                  /-
                                    🎉 no goals
                                  -/


/-- `HasKernels` represents the existence of kernels for every morphism. -/
class HasKernels : Prop where
  has_limit : ∀ {X Y : C} (f : X ⟶ Y), HasKernel f := by infer_instance


/-- `HasCokernels` represents the existence of cokernels for every morphism. -/
class HasCokernels : Prop where
  has_colimit : ∀ {X Y : C} (f : X ⟶ Y), HasCokernel f := by infer_instance


instance (priority := 100) hasKernels_of_hasEqualizers [HasEqualizers C] : HasKernels C where


instance (priority := 100) hasCokernels_of_hasCoequalizers [HasCoequalizers C] :
    HasCokernels C where


