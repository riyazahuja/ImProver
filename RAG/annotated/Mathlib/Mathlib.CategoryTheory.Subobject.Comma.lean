/-- Every subobject of a structured arrow can be projected to a subobject of the underlying
    object. -/
def projectSubobject [HasFiniteLimits C] [PreservesFiniteLimits T] {A : StructuredArrow S T} :
    Subobject A → Subobject A.right := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
    A : CategoryTheory.StructuredArrow S T
    ⊢ CategoryTheory.Subobject A → CategoryTheory.Subobject A.right
  -/
  refine Subobject.lift (fun P f hf => Subobject.mk f.right) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
    A : CategoryTheory.StructuredArrow S T
    ⊢ ∀ ⦃A_1 B : CategoryTheory.StructuredArrow S T⦄ (f : Quiver.Hom A_1 A) (g : Q …
  -/
  intro P Q f g hf hg i hi
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
    A P Q : CategoryTheory.StructuredArrow S T
    f : Quiver.Hom P A
    g : Quiver.Hom Q A
    hf : CategoryTheory.Mono f
    hg : CategoryTheory.Mono g
    i : CategoryTheory.Iso P Q
    hi : Eq (CategoryTheory.CategoryStruct.comp i.hom g) f
    ⊢ Eq ((fun P f hf => CategoryTheory.Subobject.mk f.right) P f hf) ((fun P f hf …
  -/
  refine Subobject.mk_eq_mk_of_comm _ _ ((proj S T).mapIso i) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : D
    T : CategoryTheory.Functor C D
    inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
    A P Q : CategoryTheory.StructuredArrow S T
    f : Quiver.Hom P A
    g : Quiver.Hom Q A
    hf : CategoryTheory.Mono f
    hg : CategoryTheory.Mono g
    i : CategoryTheory.Iso P Q
    hi : Eq (CategoryTheory.CategoryStruct.comp i.hom g) f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.StructuredArrow.proj …
  -/
  exact congr_arg CommaMorphism.right hi
  /-
    🎉 no goals
  -/


@[simp]
theorem projectSubobject_mk [HasFiniteLimits C] [PreservesFiniteLimits T]
    {A P : StructuredArrow S T}
    (f : P ⟶ A) [Mono f] : projectSubobject (Subobject.mk f) = Subobject.mk f.right :=
  rfl


theorem projectSubobject_factors [HasFiniteLimits C] [PreservesFiniteLimits T]
    {A : StructuredArrow S T} :
    ∀ P : Subobject A, ∃ q, q ≫ T.map (projectSubobject P).arrow = A.hom :=
  Subobject.ind _ fun P f hf =>
    ⟨P.hom ≫ T.map (Subobject.underlyingIso _).inv, by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A P : CategoryTheory.StructuredArrow S T
        f : Quiver.Hom P A
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp P …
      -/
      dsimp
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A P : CategoryTheory.StructuredArrow S T
        f : Quiver.Hom P A
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp P …
      -/
      simp [← T.map_comp]⟩
      /-
        🎉 no goals
      -/


/-- A subobject of the underlying object of a structured arrow can be lifted to a subobject of
    the structured arrow, provided that there is a morphism making the subobject into a structured
    arrow. -/
@[simp]
def liftSubobject {A : StructuredArrow S T} (P : Subobject A.right) {q}
    (hq : q ≫ T.map P.arrow = A.hom) : Subobject A :=
  Subobject.mk (homMk P.arrow hq : mk q ⟶ A)


/-- Projecting and then lifting a subobject recovers the original subobject, because there is at
    most one morphism making the projected subobject into a structured arrow. -/
theorem lift_projectSubobject [HasFiniteLimits C] [PreservesFiniteLimits T]
    {A : StructuredArrow S T} :
    ∀ (P : Subobject A) {q} (hq : q ≫ T.map (projectSubobject P).arrow = A.hom),
      liftSubobject (projectSubobject P) hq = P :=
  Subobject.ind _
    (by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A : CategoryTheory.StructuredArrow S T
        ⊢ ∀ ⦃A_1 : CategoryTheory.StructuredArrow S T⦄ (f : Quiver.Hom A_1 A) [inst :  …
      -/
      intro P f hf q hq
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A P : CategoryTheory.StructuredArrow S T
        f : Quiver.Hom P A
        hf : CategoryTheory.Mono f
        q : Quiver.Hom ((CategoryTheory.Functor.fromPUnit S).obj A.left) (T.obj (Categ …
        hq : Eq (CategoryTheory.CategoryStruct.comp q (T.map (CategoryTheory.Structure …
        ⊢ Eq (CategoryTheory.StructuredArrow.liftSubobject (CategoryTheory.StructuredA …
      -/
      fapply Subobject.mk_eq_mk_of_comm
        /-
          case i
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          S : D
          T : CategoryTheory.Functor C D
          inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
          inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
          A P : CategoryTheory.StructuredArrow S T
          f : Quiver.Hom P A
          hf : CategoryTheory.Mono f
          q : Quiver.Hom ((CategoryTheory.Functor.fromPUnit S).obj A.left) (T.obj (Categ …
          hq : Eq (CategoryTheory.CategoryStruct.comp q (T.map (CategoryTheory.Structure …
          ⊢ CategoryTheory.Iso (CategoryTheory.StructuredArrow.mk q) P
        -/
      · fapply isoMk
          /-
            case i.g
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            S : D
            T : CategoryTheory.Functor C D
            inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
            inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
            A P : CategoryTheory.StructuredArrow S T
            f : Quiver.Hom P A
            hf : CategoryTheory.Mono f
            q : Quiver.Hom ((CategoryTheory.Functor.fromPUnit S).obj A.left) (T.obj (Categ …
            hq : Eq (CategoryTheory.CategoryStruct.comp q (T.map (CategoryTheory.Structure …
            ⊢ CategoryTheory.Iso (CategoryTheory.StructuredArrow.mk q).right P.right
          -/
        · exact Subobject.underlyingIso _
          /-
            🎉 no goals
          -/
          /-
            case i.w
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            S : D
            T : CategoryTheory.Functor C D
            inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
            inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
            A P : CategoryTheory.StructuredArrow S T
            f : Quiver.Hom P A
            hf : CategoryTheory.Mono f
            q : Quiver.Hom ((CategoryTheory.Functor.fromPUnit S).obj A.left) (T.obj (Categ …
            hq : Eq (CategoryTheory.CategoryStruct.comp q (T.map (CategoryTheory.Structure …
            ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Structured …
          -/
        · exact (cancel_mono (T.map f.right)).1 (by dsimp; simpa [← T.map_comp] using hq)
          /-
            🎉 no goals
          -/
        /-
          case w
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          S : D
          T : CategoryTheory.Functor C D
          inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
          inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
          A P : CategoryTheory.StructuredArrow S T
          f : Quiver.Hom P A
          hf : CategoryTheory.Mono f
          q : Quiver.Hom ((CategoryTheory.Functor.fromPUnit S).obj A.left) (T.obj (Categ …
          hq : Eq (CategoryTheory.CategoryStruct.comp q (T.map (CategoryTheory.Structure …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.isoMk …
        -/
      · exact ext _ _ (by dsimp; simp))
        /-
          🎉 no goals
        -/


/-- If `A : S → T.obj B` is a structured arrow for `S : D` and `T : C ⥤ D`, then we can explicitly
    describe the subobjects of `A` as the subobjects `P` of `B` in `C` for which `A.hom` factors
    through the image of `P` under `T`. -/
@[simps!]
def subobjectEquiv [HasFiniteLimits C] [PreservesFiniteLimits T] (A : StructuredArrow S T) :
    Subobject A ≃o { P : Subobject A.right // ∃ q, q ≫ T.map P.arrow = A.hom } where
  toFun P := ⟨projectSubobject P, projectSubobject_factors P⟩
  invFun P := liftSubobject P.val P.prop.choose_spec
  left_inv _ := lift_projectSubobject _ _
  right_inv P := Subtype.ext (by simp only [liftSubobject, homMk_right, projectSubobject_mk,
      Subobject.mk_arrow, Subtype.coe_eta])
  map_rel_iff' := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S : D
      T : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
      A : CategoryTheory.StructuredArrow S T
      ⊢ ∀ {a b : CategoryTheory.Subobject A}, Iff (LE.le ({ toFun := fun P => ⟨Categ …
    -/
    apply Subobject.ind₂
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S : D
      T : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
      A : CategoryTheory.StructuredArrow S T
      ⊢ ∀ ⦃A_1 B : CategoryTheory.StructuredArrow S T⦄ (f : Quiver.Hom A_1 A) (g : Q …
    -/
    intro P Q f g hf hg
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S : D
      T : CategoryTheory.Functor C D
      inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
      A P Q : CategoryTheory.StructuredArrow S T
      f : Quiver.Hom P A
      g : Quiver.Hom Q A
      hf : CategoryTheory.Mono f
      hg : CategoryTheory.Mono g
      ⊢ Iff (LE.le ({ toFun := fun P => ⟨CategoryTheory.StructuredArrow.projectSubob …
    -/
    refine ⟨fun h => Subobject.mk_le_mk_of_comm ?_ ?_, fun h => ?_⟩
    · exact homMk (Subobject.ofMkLEMk _ _ h)
        ((cancel_mono (T.map g.right)).1 (by simp [← T.map_comp]))
      /-
        case h.refine_2
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A P Q : CategoryTheory.StructuredArrow S T
        f : Quiver.Hom P A
        g : Quiver.Hom Q A
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le ({ toFun := fun P => ⟨CategoryTheory.StructuredArrow.projectSubobjec …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.StructuredArrow.homMk …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/
      /-
        case h.refine_3
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A P Q : CategoryTheory.StructuredArrow S T
        f : Quiver.Hom P A
        g : Quiver.Hom Q A
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ LE.le ({ toFun := fun P => ⟨CategoryTheory.StructuredArrow.projectSubobject  …
      -/
    · refine Subobject.mk_le_mk_of_comm (Subobject.ofMkLEMk _ _ h).right ?_
      /-
        case h.refine_3
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : D
        T : CategoryTheory.Functor C D
        inst✝¹ : CategoryTheory.Limits.HasFiniteLimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteLimits T
        A P Q : CategoryTheory.StructuredArrow S T
        f : Quiver.Hom P A
        g : Quiver.Hom Q A
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLEMk f  …
      -/
      exact congr_arg CommaMorphism.right (Subobject.ofMkLEMk_comp h)
      /-
        🎉 no goals
      -/

-- These lemmas have always been bad (https://github.com/leanprover-community/mathlib4/issues/7657), but https://github.com/leanprover/lean4/pull/2644 made `simp` start noticing

/-- If `C` is well-powered and complete and `T` preserves limits, then `StructuredArrow S T` is
    well-powered. -/
instance wellPowered_structuredArrow [LocallySmall.{w} C]
    [WellPowered.{w} C] [HasFiniteLimits C] [PreservesFiniteLimits T] :
    WellPowered.{w} (StructuredArrow S T) where
  subobject_small X := small_map (subobjectEquiv X).toEquiv


/-- Every quotient of a costructured arrow can be projected to a quotient of the underlying
    object. -/
def projectQuotient [HasFiniteColimits C] [PreservesFiniteColimits S] {A : CostructuredArrow S T} :
    Subobject (op A) → Subobject (op A.left) := by
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
    A : CategoryTheory.CostructuredArrow S T
    ⊢ CategoryTheory.Subobject { unop := A } → CategoryTheory.Subobject { unop :=  …
  -/
  refine Subobject.lift (fun P f hf => Subobject.mk f.unop.left.op) ?_
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
    A : CategoryTheory.CostructuredArrow S T
    ⊢ ∀ ⦃A_1 B : Opposite (CategoryTheory.CostructuredArrow S T)⦄ (f : Quiver.Hom  …
  -/
  intro P Q f g hf hg i hi
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
    A : CategoryTheory.CostructuredArrow S T
    P Q : Opposite (CategoryTheory.CostructuredArrow S T)
    f : Quiver.Hom P { unop := A }
    g : Quiver.Hom Q { unop := A }
    hf : CategoryTheory.Mono f
    hg : CategoryTheory.Mono g
    i : CategoryTheory.Iso P Q
    hi : Eq (CategoryTheory.CategoryStruct.comp i.hom g) f
    ⊢ Eq ((fun P f hf => CategoryTheory.Subobject.mk f.unop.left.op) P f hf) ((fun …
  -/
  refine Subobject.mk_eq_mk_of_comm _ _ ((proj S T).mapIso i.unop).op (Quiver.Hom.unop_inj ?_)
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
    A : CategoryTheory.CostructuredArrow S T
    P Q : Opposite (CategoryTheory.CostructuredArrow S T)
    f : Quiver.Hom P { unop := A }
    g : Quiver.Hom Q { unop := A }
    hf : CategoryTheory.Mono f
    hg : CategoryTheory.Mono g
    i : CategoryTheory.Iso P Q
    hi : Eq (CategoryTheory.CategoryStruct.comp i.hom g) f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CostructuredArrow.pr …
  -/
  have := congr_arg Quiver.Hom.unop hi
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
    inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
    A : CategoryTheory.CostructuredArrow S T
    P Q : Opposite (CategoryTheory.CostructuredArrow S T)
    f : Quiver.Hom P { unop := A }
    g : Quiver.Hom Q { unop := A }
    hf : CategoryTheory.Mono f
    hg : CategoryTheory.Mono g
    i : CategoryTheory.Iso P Q
    hi : Eq (CategoryTheory.CategoryStruct.comp i.hom g) f
    this : Eq (CategoryTheory.CategoryStruct.comp i.hom g).unop f.unop
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CostructuredArrow.pr …
  -/
  simpa using congr_arg CommaMorphism.left this
  /-
    🎉 no goals
  -/


@[simp]
theorem projectQuotient_mk [HasFiniteColimits C] [PreservesFiniteColimits S]
    {A : CostructuredArrow S T}
    {P : (CostructuredArrow S T)ᵒᵖ} (f : P ⟶ op A) [Mono f] :
    projectQuotient (Subobject.mk f) = Subobject.mk f.unop.left.op :=
  rfl


theorem projectQuotient_factors [HasFiniteColimits C] [PreservesFiniteColimits S]
    {A : CostructuredArrow S T} :
    ∀ P : Subobject (op A), ∃ q, S.map (projectQuotient P).arrow.unop ≫ q = A.hom :=
  Subobject.ind _ fun P f hf =>
    ⟨S.map (Subobject.underlyingIso _).unop.inv ≫ P.unop.hom, by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.CostructuredAr …
      -/
      dsimp
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Subobject.mk f …
      -/
      rw [← Category.assoc, ← S.map_comp, ← unop_comp]
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        hf : CategoryTheory.Mono f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.CategoryStruct …
      -/
      simp⟩
      /-
        🎉 no goals
      -/


/-- A quotient of the underlying object of a costructured arrow can be lifted to a quotient of
    the costructured arrow, provided that there is a morphism making the quotient into a
    costructured arrow. -/
@[simp]
def liftQuotient {A : CostructuredArrow S T} (P : Subobject (op A.left)) {q}
    (hq : S.map P.arrow.unop ≫ q = A.hom) : Subobject (op A) :=
  Subobject.mk (homMk P.arrow.unop hq : A ⟶ mk q).op


/-- Technical lemma for `lift_projectQuotient`. -/
@[simp]
theorem unop_left_comp_underlyingIso_hom_unop {A : CostructuredArrow S T}
    {P : (CostructuredArrow S T)ᵒᵖ} (f : P ⟶ op A) [Mono f.unop.left.op] :
    f.unop.left ≫ (Subobject.underlyingIso f.unop.left.op).hom.unop =
      (Subobject.mk f.unop.left.op).arrow.unop := by
  conv_lhs =>
    congr
    rw [← Quiver.Hom.unop_op f.unop.left]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    A : CategoryTheory.CostructuredArrow S T
    P : Opposite (CategoryTheory.CostructuredArrow S T)
    f : Quiver.Hom P { unop := A }
    inst✝ : CategoryTheory.Mono f.unop.left.op
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f.unop.left.op.unop (CategoryTheory.S …
  -/
  rw [← unop_comp, Subobject.underlyingIso_hom_comp_eq_mk]
  /-
    🎉 no goals
  -/


/-- Projecting and then lifting a quotient recovers the original quotient, because there is at most
    one morphism making the projected quotient into a costructured arrow. -/
theorem lift_projectQuotient [HasFiniteColimits C] [PreservesFiniteColimits S]
    {A : CostructuredArrow S T} :
    ∀ (P : Subobject (op A)) {q} (hq : S.map (projectQuotient P).arrow.unop ≫ q = A.hom),
      liftQuotient (projectQuotient P) hq = P :=
  Subobject.ind _
    (by
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        ⊢ ∀ ⦃A_1 : Opposite (CategoryTheory.CostructuredArrow S T)⦄ (f : Quiver.Hom A_ …
      -/
      intro P f hf q hq
      /-
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        hf : CategoryTheory.Mono f
        q : Quiver.Hom (S.obj (Opposite.unop (CategoryTheory.Subobject.underlying.obj  …
        hq : Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Costructure …
        ⊢ Eq (CategoryTheory.CostructuredArrow.liftQuotient (CategoryTheory.Costructur …
      -/
      fapply Subobject.mk_eq_mk_of_comm
        /-
          case i
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          S : CategoryTheory.Functor C D
          T : D
          inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
          inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
          A : CategoryTheory.CostructuredArrow S T
          P : Opposite (CategoryTheory.CostructuredArrow S T)
          f : Quiver.Hom P { unop := A }
          hf : CategoryTheory.Mono f
          q : Quiver.Hom (S.obj (Opposite.unop (CategoryTheory.Subobject.underlying.obj  …
          hq : Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Costructure …
          ⊢ CategoryTheory.Iso { unop := CategoryTheory.CostructuredArrow.mk q } P
        -/
      · refine (Iso.op (isoMk ?_ ?_) : _ ≅ op (unop P))
          /-
            case i.refine_1
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor C D
            T : D
            inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
            inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
            A : CategoryTheory.CostructuredArrow S T
            P : Opposite (CategoryTheory.CostructuredArrow S T)
            f : Quiver.Hom P { unop := A }
            hf : CategoryTheory.Mono f
            q : Quiver.Hom (S.obj (Opposite.unop (CategoryTheory.Subobject.underlying.obj  …
            hq : Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Costructure …
            ⊢ CategoryTheory.Iso (Opposite.unop P).left (CategoryTheory.CostructuredArrow. …
          -/
        · exact (Subobject.underlyingIso f.unop.left.op).unop
          /-
            🎉 no goals
          -/
          /-
            case i.refine_2
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor C D
            T : D
            inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
            inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
            A : CategoryTheory.CostructuredArrow S T
            P : Opposite (CategoryTheory.CostructuredArrow S T)
            f : Quiver.Hom P { unop := A }
            hf : CategoryTheory.Mono f
            q : Quiver.Hom (S.obj (Opposite.unop (CategoryTheory.Subobject.underlying.obj  …
            hq : Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Costructure …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Subobject.unde …
          -/
        · refine (cancel_epi (S.map f.unop.left)).1 ?_
          /-
            case i.refine_2
            C : Type u₁
            inst✝³ : CategoryTheory.Category.{v₁, u₁} C
            D : Type u₂
            inst✝² : CategoryTheory.Category.{v₂, u₂} D
            S : CategoryTheory.Functor C D
            T : D
            inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
            inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
            A : CategoryTheory.CostructuredArrow S T
            P : Opposite (CategoryTheory.CostructuredArrow S T)
            f : Quiver.Hom P { unop := A }
            hf : CategoryTheory.Mono f
            q : Quiver.Hom (S.obj (Opposite.unop (CategoryTheory.Subobject.underlying.obj  …
            hq : Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Costructure …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map f.unop.left) (CategoryTheory.C …
          -/
          simpa [← Category.assoc, ← S.map_comp] using hq
          /-
            🎉 no goals
          -/
        /-
          case w
          C : Type u₁
          inst✝³ : CategoryTheory.Category.{v₁, u₁} C
          D : Type u₂
          inst✝² : CategoryTheory.Category.{v₂, u₂} D
          S : CategoryTheory.Functor C D
          T : D
          inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
          inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
          A : CategoryTheory.CostructuredArrow S T
          P : Opposite (CategoryTheory.CostructuredArrow S T)
          f : Quiver.Hom P { unop := A }
          hf : CategoryTheory.Mono f
          q : Quiver.Hom (S.obj (Opposite.unop (CategoryTheory.Subobject.underlying.obj  …
          hq : Eq (CategoryTheory.CategoryStruct.comp (S.map (CategoryTheory.Costructure …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CostructuredArrow.iso …
        -/
      · exact Quiver.Hom.unop_inj (by aesop_cat))
        /-
          🎉 no goals
        -/


/-- Technical lemma for `quotientEquiv`. -/
theorem unop_left_comp_ofMkLEMk_unop {A : CostructuredArrow S T} {P Q : (CostructuredArrow S T)ᵒᵖ}
    {f : P ⟶ op A} {g : Q ⟶ op A} [Mono f.unop.left.op] [Mono g.unop.left.op]
    (h : Subobject.mk f.unop.left.op ≤ Subobject.mk g.unop.left.op) :
    g.unop.left ≫ (Subobject.ofMkLEMk f.unop.left.op g.unop.left.op h).unop = f.unop.left := by
  conv_lhs =>
    congr
    rw [← Quiver.Hom.unop_op g.unop.left]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    A : CategoryTheory.CostructuredArrow S T
    P Q : Opposite (CategoryTheory.CostructuredArrow S T)
    f : Quiver.Hom P { unop := A }
    g : Quiver.Hom Q { unop := A }
    inst✝¹ : CategoryTheory.Mono f.unop.left.op
    inst✝ : CategoryTheory.Mono g.unop.left.op
    h : LE.le (CategoryTheory.Subobject.mk f.unop.left.op) (CategoryTheory.Subobje …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp g.unop.left.op.unop (CategoryTheory.S …
  -/
  rw [← unop_comp]
  /-
    C : Type u₁
    inst✝³ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    S : CategoryTheory.Functor C D
    T : D
    A : CategoryTheory.CostructuredArrow S T
    P Q : Opposite (CategoryTheory.CostructuredArrow S T)
    f : Quiver.Hom P { unop := A }
    g : Quiver.Hom Q { unop := A }
    inst✝¹ : CategoryTheory.Mono f.unop.left.op
    inst✝ : CategoryTheory.Mono g.unop.left.op
    h : LE.le (CategoryTheory.Subobject.mk f.unop.left.op) (CategoryTheory.Subobje …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLEMk f. …
  -/
  simp only [Subobject.ofMkLEMk_comp, Quiver.Hom.unop_op]
  /-
    🎉 no goals
  -/


/-- If `A : S.obj B ⟶ T` is a costructured arrow for `S : C ⥤ D` and `T : D`, then we can
    explicitly describe the quotients of `A` as the quotients `P` of `B` in `C` for which `A.hom`
    factors through the image of `P` under `S`. -/
def quotientEquiv [HasFiniteColimits C] [PreservesFiniteColimits S] (A : CostructuredArrow S T) :
    Subobject (op A) ≃o { P : Subobject (op A.left) // ∃ q, S.map P.arrow.unop ≫ q = A.hom } where
  toFun P := ⟨projectQuotient P, projectQuotient_factors P⟩
  invFun P := liftQuotient P.val P.prop.choose_spec
  left_inv _ := lift_projectQuotient _ _
  right_inv P := Subtype.ext (by simp only [liftQuotient, Quiver.Hom.unop_op, homMk_left,
      Quiver.Hom.op_unop, projectQuotient_mk, Subobject.mk_arrow])
  map_rel_iff' := by
    /-
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor C D
      T : D
      inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
      A : CategoryTheory.CostructuredArrow S T
      ⊢ ∀ {a b : CategoryTheory.Subobject { unop := A }}, Iff (LE.le ({ toFun := fun …
    -/
    apply Subobject.ind₂
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor C D
      T : D
      inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
      A : CategoryTheory.CostructuredArrow S T
      ⊢ ∀ ⦃A_1 B : Opposite (CategoryTheory.CostructuredArrow S T)⦄ (f : Quiver.Hom  …
    -/
    intro P Q f g hf hg
    /-
      case h
      C : Type u₁
      inst✝³ : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      S : CategoryTheory.Functor C D
      T : D
      inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
      inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
      A : CategoryTheory.CostructuredArrow S T
      P Q : Opposite (CategoryTheory.CostructuredArrow S T)
      f : Quiver.Hom P { unop := A }
      g : Quiver.Hom Q { unop := A }
      hf : CategoryTheory.Mono f
      hg : CategoryTheory.Mono g
      ⊢ Iff (LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuo …
    -/
    refine ⟨fun h => Subobject.mk_le_mk_of_comm ?_ ?_, fun h => ?_⟩
      /-
        case h.refine_1
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuotie …
        ⊢ Quiver.Hom P Q
      -/
    · refine (homMk (Subobject.ofMkLEMk _ _ h).unop ((cancel_epi (S.map g.unop.left)).1 ?_)).op
      /-
        case h.refine_1
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuotie …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map g.unop.left) (CategoryTheory.C …
      -/
      dsimp
      simp only [← S.map_comp_assoc, unop_left_comp_ofMkLEMk_unop, unop_op, CommaMorphism.w,
        Functor.const_obj_obj, right_eq_id, Functor.const_obj_map, Category.comp_id]
      /-
        case h.refine_2
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuotie …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CostructuredArrow.hom …
      -/
    · apply Quiver.Hom.unop_inj
      /-
        case h.refine_2.a
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuotie …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CostructuredArrow.hom …
      -/
      ext
      /-
        case h.refine_2.a.h
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuotie …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CostructuredArrow.hom …
      -/
      exact unop_left_comp_ofMkLEMk_unop _
      /-
        🎉 no goals
      -/
      /-
        case h.refine_3
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ LE.le ({ toFun := fun P => ⟨CategoryTheory.CostructuredArrow.projectQuotient …
      -/
    · refine Subobject.mk_le_mk_of_comm (Subobject.ofMkLEMk _ _ h).unop.left.op ?_
      /-
        case h.refine_3
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLEMk f  …
      -/
      refine Quiver.Hom.unop_inj ?_
      /-
        case h.refine_3
        C : Type u₁
        inst✝³ : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝² : CategoryTheory.Category.{v₂, u₂} D
        S : CategoryTheory.Functor C D
        T : D
        inst✝¹ : CategoryTheory.Limits.HasFiniteColimits C
        inst✝ : CategoryTheory.Limits.PreservesFiniteColimits S
        A : CategoryTheory.CostructuredArrow S T
        P Q : Opposite (CategoryTheory.CostructuredArrow S T)
        f : Quiver.Hom P { unop := A }
        g : Quiver.Hom Q { unop := A }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subobject.ofMkLEMk f  …
      -/
      have := congr_arg Quiver.Hom.unop (Subobject.ofMkLEMk_comp h)
      simpa only [unop_op, Functor.id_obj, Functor.const_obj_obj, MonoOver.mk'_obj, Over.mk_left,
        MonoOver.mk'_arrow, unop_comp, Quiver.Hom.unop_op, comp_left]
          using congr_arg CommaMorphism.left this


/-- If `C` is well-copowered and cocomplete and `S` preserves colimits, then
    `CostructuredArrow S T` is well-copowered. -/
instance well_copowered_costructuredArrow [LocallySmall.{w} C] [WellPowered.{w} Cᵒᵖ]
    [HasFiniteColimits C] [PreservesFiniteColimits S] :
    WellPowered.{w} (CostructuredArrow S T)ᵒᵖ where
  subobject_small X := small_map (quotientEquiv (unop X)).toEquiv


