lemma hom_comp_singleFunctor_map_shift [HasDerivedCategory.{w'} C]
    {X Y Z : C} {n : ℕ} (x : Ext X Y n) (f : Y ⟶ Z) :
    x.hom ≫ ((DerivedCategory.singleFunctor C 0).map f)⟦(n : ℤ)⟧' =
      (x.comp (mk₀ f) (add_zero n)).hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    inst✝ : HasDerivedCategory C
    X Y Z : C
    n : Nat
    x : CategoryTheory.Abelian.Ext X Y n
    f : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom ((CategoryTheory.shiftFunctor ( …
  -/
  simp only [comp_hom, mk₀_hom, ShiftedHom.comp_mk₀]
  /-
    🎉 no goals
  -/


lemma preadditiveCoyoneda_homologySequenceδ_singleTriangle_apply
    [HasDerivedCategory.{w'} C] {X : C} {n₀ : ℕ} (x : Ext X S.X₃ n₀)
    {n₁ : ℕ} (h : n₀ + 1 = n₁) :
    (preadditiveCoyoneda.obj (op ((singleFunctor C 0).obj X))).homologySequenceδ
                                  /-
                                    C : Type u
                                    inst✝³ : CategoryTheory.Category.{v, u} C
                                    inst✝² : CategoryTheory.Abelian C
                                    inst✝¹ : CategoryTheory.HasExt C
                                    X✝ : C
                                    S : CategoryTheory.ShortComplex C
                                    hS : S.ShortExact
                                    inst✝ : HasDerivedCategory C
                                    X : C
                                    n₀ : Nat
                                    x : CategoryTheory.Abelian.Ext X S.X₃ n₀
                                    n₁ : Nat
                                    h : Eq (HAdd.hAdd n₀ 1) n₁
                                    ⊢ Eq (HAdd.hAdd (↑n₀) 1) ↑n₁
                                  -/
      hS.singleTriangle n₀ n₁ (by omega) x.hom =
                                  /-
                                    🎉 no goals
                                  -/
        (x.comp hS.extClass h).hom := by
  rw [Pretriangulated.preadditiveCoyoneda_homologySequenceδ_apply,
    comp_hom, hS.extClass_hom, ShiftedHom.comp]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    X : C
    n₀ : Nat
    x : CategoryTheory.Abelian.Ext X S.X₃ n₀
    n₁ : Nat
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x.hom (CategoryTheory.CategoryStruct. …
  -/
  rfl
  /-
    🎉 no goals
  -/


include hS in
/-- Alternative formulation of `covariant_sequence_exact₂` -/
lemma covariant_sequence_exact₂' (n : ℕ) :
    (ShortComplex.mk (AddCommGrp.ofHom ((mk₀ S.f).postcomp X (add_zero n)))
      (AddCommGrp.ofHom ((mk₀ S.g).postcomp X (add_zero n))) (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.Ab …
        -/
        ext x
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n : Nat
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₁ n))
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.A …
        -/
        dsimp [AddCommGrp.ofHom]
        simp only [comp_assoc_of_third_deg_zero, mk₀_comp_mk₀, ShortComplex.zero, mk₀_zero,
          comp_zero]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n : Nat
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₁ n))
          ⊢ Eq 0 (0 x)
        -/
        rfl)).Exact := by
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n : Nat
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  letI := HasDerivedCategory.standard C
  have := (preadditiveCoyoneda.obj (op ((singleFunctor C 0).obj X))).homologySequence_exact₂ _
    (hS.singleTriangle_distinguished) n
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n : Nat
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : (CategoryTheory.ShortComplex.mk (((CategoryTheory.preadditiveCoyoneda.o …
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  rw [ShortComplex.ab_exact_iff_function_exact] at this ⊢
  apply Function.Exact.of_ladder_addEquiv_of_exact' (e₁ := Ext.homAddEquiv)
    (e₂ := Ext.homAddEquiv) (e₃ := Ext.homAddEquiv) (H := this)
  /-
    case comm₁₂
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n : Nat
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
    ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.prea …
  -/
  all_goals ext x; apply hom_comp_singleFunctor_map_shift (C := C)
  /-
    🎉 no goals
  -/


/-- Alternative formulation of `covariant_sequence_exact₃` -/
lemma covariant_sequence_exact₃' :
    (ShortComplex.mk (AddCommGrp.ofHom ((mk₀ S.g).postcomp X (add_zero n₀)))
      (AddCommGrp.ofHom (hS.extClass.postcomp X h)) (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.Ab …
        -/
        ext x
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₂ n₀))
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.A …
        -/
        dsimp [AddCommGrp.ofHom]
        simp only [comp_assoc_of_second_deg_zero, ShortComplex.ShortExact.comp_extClass,
          comp_zero]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₂ n₀))
          ⊢ Eq 0 (0 x)
        -/
        rfl)).Exact := by
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  letI := HasDerivedCategory.standard C
  have := (preadditiveCoyoneda.obj (op ((singleFunctor C 0).obj X))).homologySequence_exact₃ _
    (hS.singleTriangle_distinguished) n₀ n₁ (by omega)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd n₀ 1) n₁
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : (CategoryTheory.ShortComplex.mk (((CategoryTheory.preadditiveCoyoneda.o …
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  rw [ShortComplex.ab_exact_iff_function_exact] at this ⊢
  apply Function.Exact.of_ladder_addEquiv_of_exact' (e₁ := Ext.homAddEquiv)
    (e₂ := Ext.homAddEquiv) (e₃ := Ext.homAddEquiv) (H := this)
    /-
      case comm₁₂
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd n₀ 1) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.prea …
    -/
  · ext x; apply hom_comp_singleFunctor_map_shift (C := C)
           /-
             🎉 no goals
           -/
    /-
      case comm₂₃
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd n₀ 1) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.prea …
    -/
  · ext x
    /-
      case comm₂₃.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd n₀ 1) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
      x : CategoryTheory.Abelian.Ext X S.X₃ n₀
      ⊢ Eq ((AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.pre …
    -/
    exact preadditiveCoyoneda_homologySequenceδ_singleTriangle_apply hS x h
    /-
      🎉 no goals
    -/


/-- Alternative formulation of `covariant_sequence_exact₁` -/
lemma covariant_sequence_exact₁' :
    (ShortComplex.mk
      (AddCommGrp.ofHom (hS.extClass.postcomp X h))
      (AddCommGrp.ofHom ((mk₀ S.f).postcomp X (add_zero n₁))) (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom (hS.extClass.postco …
        -/
        ext x
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₃ n₀))
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom (hS.extClass.postc …
        -/
        dsimp [AddCommGrp.ofHom]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₃ n₀))
          ⊢ Eq ((CategoryTheory.Abelian.Ext.comp x hS.extClass h).comp (CategoryTheory.A …
        -/
        simp only [comp_assoc_of_third_deg_zero, ShortComplex.ShortExact.extClass_comp, comp_zero]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          X : C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd n₀ 1) n₁
          x : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext X S.X₃ n₀))
          ⊢ Eq 0 (0 x)
        -/
        rfl)).Exact := by
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.postcomp X h) …
  -/
  letI := HasDerivedCategory.standard C
  have := (preadditiveCoyoneda.obj (op ((singleFunctor C 0).obj X))).homologySequence_exact₁ _
    (hS.singleTriangle_distinguished) n₀ n₁ (by omega)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd n₀ 1) n₁
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : (CategoryTheory.ShortComplex.mk ((CategoryTheory.preadditiveCoyoneda.ob …
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.postcomp X h) …
  -/
  rw [ShortComplex.ab_exact_iff_function_exact] at this ⊢
  apply Function.Exact.of_ladder_addEquiv_of_exact' (e₁ := Ext.homAddEquiv)
    (e₂ := Ext.homAddEquiv) (e₃ := Ext.homAddEquiv) (H := this)
    /-
      case comm₁₂
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd n₀ 1) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk ((CategoryTheory.preadd …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk ((CategoryTheory.pread …
    -/
  · ext x
    /-
      case comm₁₂.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd n₀ 1) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk ((CategoryTheory.preadd …
      x : CategoryTheory.Abelian.Ext X S.X₃ n₀
      ⊢ Eq ((AddMonoidHom.comp (CategoryTheory.ShortComplex.mk ((CategoryTheory.prea …
    -/
    exact preadditiveCoyoneda_homologySequenceδ_singleTriangle_apply hS x h
    /-
      🎉 no goals
    -/
    /-
      case comm₂₃
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd n₀ 1) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk ((CategoryTheory.preadd …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk ((CategoryTheory.pread …
    -/
  · ext x; apply hom_comp_singleFunctor_map_shift (C := C)
           /-
             🎉 no goals
           -/


/-- Given a short exact short complex `S` in an abelian category `C` and an object `X : C`,
this is the long exact sequence
`Ext X S.X₁ n₀ → Ext X S.X₂ n₀ → Ext X S.X₃ n₀ → Ext X S.X₁ n₁ → Ext X S.X₂ n₁ → Ext X S.X₃ n₁`
when `n₀ + 1 = n₁` -/
noncomputable def covariantSequence : ComposableArrows AddCommGrp.{w} 5 :=
  mk₅ (AddCommGrp.ofHom ((mk₀ S.f).postcomp X (add_zero n₀)))
    (AddCommGrp.ofHom ((mk₀ S.g).postcomp X (add_zero n₀)))
    (AddCommGrp.ofHom (hS.extClass.postcomp X h))
    (AddCommGrp.ofHom ((mk₀ S.f).postcomp X (add_zero n₁)))
    (AddCommGrp.ofHom ((mk₀ S.g).postcomp X (add_zero n₁)))


lemma covariantSequence_exact :
    (covariantSequence X hS n₀ n₁ h).Exact :=
  exact_of_δ₀ (covariant_sequence_exact₂' X hS n₀).exact_toComposableArrows
    (exact_of_δ₀ (covariant_sequence_exact₃' X hS n₀ n₁ h).exact_toComposableArrows
      (exact_of_δ₀ (covariant_sequence_exact₁' X hS n₀ n₁ h).exact_toComposableArrows
        (covariant_sequence_exact₂' X hS n₁).exact_toComposableArrows))


lemma covariant_sequence_exact₁ {n₁ : ℕ} (x₁ : Ext X S.X₁ n₁)
    (hx₁ : x₁.comp (mk₀ S.f) (add_zero n₁) = 0) {n₀ : ℕ} (hn₀ : n₀ + 1 = n₁) :
    ∃ (x₃ : Ext X S.X₃ n₀), x₃.comp hS.extClass hn₀ = x₁ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₁ : Nat
    x₁ : CategoryTheory.Abelian.Ext X S.X₁ n₁
    hx₁ : Eq (x₁.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) 0
    n₀ : Nat
    hn₀ : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Exists fun x₃ => Eq (x₃.comp hS.extClass hn₀) x₁
  -/
  have := covariant_sequence_exact₁' X hS n₀ n₁ hn₀
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₁ : Nat
    x₁ : CategoryTheory.Abelian.Ext X S.X₁ n₁
    hx₁ : Eq (x₁.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) 0
    n₀ : Nat
    hn₀ : Eq (HAdd.hAdd n₀ 1) n₁
    this : (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.postcomp …
    ⊢ Exists fun x₃ => Eq (x₃.comp hS.extClass hn₀) x₁
  -/
  rw [ShortComplex.ab_exact_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₁ : Nat
    x₁ : CategoryTheory.Abelian.Ext X S.X₁ n₁
    hx₁ : Eq (x₁.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) 0
    n₀ : Nat
    hn₀ : Eq (HAdd.hAdd n₀ 1) n₁
    this : ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass …
    ⊢ Exists fun x₃ => Eq (x₃.comp hS.extClass hn₀) x₁
  -/
  exact this x₁ hx₁
  /-
    🎉 no goals
  -/


include hS in
lemma covariant_sequence_exact₂ {n : ℕ} (x₂ : Ext X S.X₂ n)
    (hx₂ : x₂.comp (mk₀ S.g) (add_zero n) = 0) :
    ∃ (x₁ : Ext X S.X₁ n), x₁.comp (mk₀ S.f) (add_zero n) = x₂ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n : Nat
    x₂ : CategoryTheory.Abelian.Ext X S.X₂ n
    hx₂ : Eq (x₂.comp (CategoryTheory.Abelian.Ext.mk₀ S.g) ⋯) 0
    ⊢ Exists fun x₁ => Eq (x₁.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) x₂
  -/
  have := covariant_sequence_exact₂' X hS n
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n : Nat
    x₂ : CategoryTheory.Abelian.Ext X S.X₂ n
    hx₂ : Eq (x₂.comp (CategoryTheory.Abelian.Ext.mk₀ S.g) ⋯) 0
    this : (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abel …
    ⊢ Exists fun x₁ => Eq (x₁.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) x₂
  -/
  rw [ShortComplex.ab_exact_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n : Nat
    x₂ : CategoryTheory.Abelian.Ext X S.X₂ n
    hx₂ : Eq (x₂.comp (CategoryTheory.Abelian.Ext.mk₀ S.g) ⋯) 0
    this : ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTh …
    ⊢ Exists fun x₁ => Eq (x₁.comp (CategoryTheory.Abelian.Ext.mk₀ S.f) ⋯) x₂
  -/
  exact this x₂ hx₂
  /-
    🎉 no goals
  -/


lemma covariant_sequence_exact₃ {n₀ : ℕ} (x₃ : Ext X S.X₃ n₀) {n₁ : ℕ} (hn₁ : n₀ + 1 = n₁)
    (hx₃ : x₃.comp hS.extClass hn₁ = 0) :
    ∃ (x₂ : Ext X S.X₂ n₀), x₂.comp (mk₀ S.g) (add_zero n₀) = x₃ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ : Nat
    x₃ : CategoryTheory.Abelian.Ext X S.X₃ n₀
    n₁ : Nat
    hn₁ : Eq (HAdd.hAdd n₀ 1) n₁
    hx₃ : Eq (x₃.comp hS.extClass hn₁) 0
    ⊢ Exists fun x₂ => Eq (x₂.comp (CategoryTheory.Abelian.Ext.mk₀ S.g) ⋯) x₃
  -/
  have := covariant_sequence_exact₃' X hS n₀ n₁ hn₁
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ : Nat
    x₃ : CategoryTheory.Abelian.Ext X S.X₃ n₀
    n₁ : Nat
    hn₁ : Eq (HAdd.hAdd n₀ 1) n₁
    hx₃ : Eq (x₃.comp hS.extClass hn₁) 0
    this : (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abel …
    ⊢ Exists fun x₂ => Eq (x₂.comp (CategoryTheory.Abelian.Ext.mk₀ S.g) ⋯) x₃
  -/
  rw [ShortComplex.ab_exact_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X : C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    n₀ : Nat
    x₃ : CategoryTheory.Abelian.Ext X S.X₃ n₀
    n₁ : Nat
    hn₁ : Eq (HAdd.hAdd n₀ 1) n₁
    hx₃ : Eq (x₃.comp hS.extClass hn₁) 0
    this : ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTh …
    ⊢ Exists fun x₂ => Eq (x₂.comp (CategoryTheory.Abelian.Ext.mk₀ S.g) ⋯) x₃
  -/
  exact this x₃ hx₃
  /-
    🎉 no goals
  -/


lemma singleFunctor_map_comp_hom [HasDerivedCategory.{w'} C]
    {X Y Z : C} (f : X ⟶ Y) {n : ℕ} (x : Ext Y Z n) :
    (DerivedCategory.singleFunctor C 0).map f ≫ x.hom =
      ((mk₀ f).comp x (zero_add n)).hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    inst✝ : HasDerivedCategory C
    X Y Z : C
    f : Quiver.Hom X Y
    n : Nat
    x : CategoryTheory.Abelian.Ext Y Z n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((DerivedCategory.singleFunctor C 0). …
  -/
  simp only [comp_hom, mk₀_hom, ShiftedHom.mk₀_comp]
  /-
    🎉 no goals
  -/


lemma preadditiveYoneda_homologySequenceδ_singleTriangle_apply
    [HasDerivedCategory.{w'} C] {Y : C} {n₀ : ℕ} (x : Ext S.X₁ Y n₀)
    {n₁ : ℕ} (h : 1 + n₀ = n₁) :
    (preadditiveYoneda.obj ((singleFunctor C 0).obj Y)).homologySequenceδ
                                                                               /-
                                                                                 C : Type u
                                                                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                                                                 inst✝² : CategoryTheory.Abelian C
                                                                                 inst✝¹ : CategoryTheory.HasExt C
                                                                                 S : CategoryTheory.ShortComplex C
                                                                                 hS : S.ShortExact
                                                                                 Y✝ : C
                                                                                 inst✝ : HasDerivedCategory C
                                                                                 Y : C
                                                                                 n₀ : Nat
                                                                                 x : CategoryTheory.Abelian.Ext S.X₁ Y n₀
                                                                                 n₁ : Nat
                                                                                 h : Eq (HAdd.hAdd 1 n₀) n₁
                                                                                 ⊢ Eq (HAdd.hAdd (↑n₀) 1) ↑n₁
                                                                               -/
      ((triangleOpEquivalence _).functor.obj (op hS.singleTriangle)) n₀ n₁ (by omega) x.hom =
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
      (hS.extClass.comp x h).hom := by
  rw [preadditiveYoneda_homologySequenceδ_apply,
    comp_hom, hS.extClass_hom, ShiftedHom.comp]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    inst✝ : HasDerivedCategory C
    Y : C
    n₀ : Nat
    x : CategoryTheory.Abelian.Ext S.X₁ Y n₀
    n₁ : Nat
    h : Eq (HAdd.hAdd 1 n₀) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp hS.singleTriangle.mor₃ (CategoryTheor …
  -/
  rfl
  /-
    🎉 no goals
  -/


include hS in
/-- Alternative formulation of `contravariant_sequence_exact₂` -/
lemma contravariant_sequence_exact₂' (n : ℕ) :
    (ShortComplex.mk (AddCommGrp.ofHom ((mk₀ S.g).precomp Y (zero_add n)))
      (AddCommGrp.ofHom ((mk₀ S.f).precomp Y (zero_add n))) (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n : Nat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.Ab …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n : Nat
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₃ Y n))
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.A …
        -/
        dsimp [AddCommGrp.ofHom]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n : Nat
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₃ Y n))
          ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp ((CategoryTheory.Abelian.Ext.m …
        -/
        simp only [mk₀_comp_mk₀_assoc, ShortComplex.zero, mk₀_zero, zero_comp]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n : Nat
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₃ Y n))
          ⊢ Eq 0 (0 x✝)
        -/
        rfl)).Exact := by
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n : Nat
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  letI := HasDerivedCategory.standard C
  have := (preadditiveYoneda.obj ((singleFunctor C 0).obj Y)).homologySequence_exact₂ _
    (op_distinguished _ hS.singleTriangle_distinguished) n
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n : Nat
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : (CategoryTheory.ShortComplex.mk (((CategoryTheory.preadditiveYoneda.obj …
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  rw [ShortComplex.ab_exact_iff_function_exact] at this ⊢
  apply Function.Exact.of_ladder_addEquiv_of_exact' (e₁ := Ext.homAddEquiv)
    (e₂ := Ext.homAddEquiv) (e₃ := Ext.homAddEquiv) (H := this)
  /-
    case comm₁₂
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n : Nat
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
    ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.prea …
  -/
  all_goals ext; apply singleFunctor_map_comp_hom (C := C)
  /-
    🎉 no goals
  -/


/-- Alternative formulation of `contravariant_sequence_exact₁` -/
lemma contravariant_sequence_exact₁' :
    (ShortComplex.mk (AddCommGrp.ofHom (((mk₀ S.f).precomp Y (zero_add n₀))))
      (AddCommGrp.ofHom (hS.extClass.precomp Y h)) (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.Ab …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₂ Y n₀))
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.A …
        -/
        dsimp [AddCommGrp.ofHom]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₂ Y n₀))
          ⊢ Eq (hS.extClass.comp ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x✝ ⋯) h) (0  …
        -/
        simp only [ShortComplex.ShortExact.extClass_comp_assoc]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₂ Y n₀))
          ⊢ Eq 0 (0 x✝)
        -/
        rfl)).Exact := by
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd 1 n₀) n₁
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  letI := HasDerivedCategory.standard C
  have := (preadditiveYoneda.obj ((singleFunctor C 0).obj Y)).homologySequence_exact₃ _
    (op_distinguished _ hS.singleTriangle_distinguished) n₀ n₁ (by omega)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd 1 n₀) n₁
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : (CategoryTheory.ShortComplex.mk (((CategoryTheory.preadditiveYoneda.obj …
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abelian.E …
  -/
  rw [ShortComplex.ab_exact_iff_function_exact] at this ⊢
  apply Function.Exact.of_ladder_addEquiv_of_exact' (e₁ := Ext.homAddEquiv)
    (e₂ := Ext.homAddEquiv) (e₃ := Ext.homAddEquiv) (H := this)
    /-
      case comm₁₂
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      Y : C
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd 1 n₀) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.prea …
    -/
  · ext; apply singleFunctor_map_comp_hom (C := C)
         /-
           🎉 no goals
         -/
    /-
      case comm₂₃
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      Y : C
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd 1 n₀) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk (((CategoryTheory.pread …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk (((CategoryTheory.prea …
    -/
  · ext; dsimp; apply preadditiveYoneda_homologySequenceδ_singleTriangle_apply
                /-
                  🎉 no goals
                -/


/-- Alternative formulation of `contravariant_sequence_exact₃` -/
lemma contravariant_sequence_exact₃' :
    (ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.precomp Y h))
      (AddCommGrp.ofHom (((mk₀ S.g).precomp Y (zero_add n₁)))) (by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom (hS.extClass.precom …
        -/
        ext
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₁ Y n₀))
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom (hS.extClass.preco …
        -/
        dsimp [AddCommGrp.ofHom]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₁ Y n₀))
          ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp (hS.extClass.comp x✝ h) ⋯) (0  …
        -/
        simp only [ShortComplex.ShortExact.comp_extClass_assoc]
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          S : CategoryTheory.ShortComplex C
          hS : S.ShortExact
          Y : C
          n₀ n₁ : Nat
          h : Eq (HAdd.hAdd 1 n₀) n₁
          x✝ : ↑(AddCommGrp.of (CategoryTheory.Abelian.Ext S.X₁ Y n₀))
          ⊢ Eq 0 (0 x✝)
        -/
        rfl)).Exact := by
        /-
          🎉 no goals
        -/
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd 1 n₀) n₁
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.precomp Y h)) …
  -/
  letI := HasDerivedCategory.standard C
  have := (preadditiveYoneda.obj ((singleFunctor C 0).obj Y)).homologySequence_exact₁ _
    (op_distinguished _ hS.singleTriangle_distinguished) n₀ n₁ (by omega)
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ n₁ : Nat
    h : Eq (HAdd.hAdd 1 n₀) n₁
    this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
    this : (CategoryTheory.ShortComplex.mk ((CategoryTheory.preadditiveYoneda.obj  …
    ⊢ (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.precomp Y h)) …
  -/
  rw [ShortComplex.ab_exact_iff_function_exact] at this ⊢
  apply Function.Exact.of_ladder_addEquiv_of_exact' (e₁ := Ext.homAddEquiv)
    (e₂ := Ext.homAddEquiv) (e₃ := Ext.homAddEquiv) (H := this)
    /-
      case comm₁₂
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      Y : C
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd 1 n₀) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk ((CategoryTheory.preadd …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk ((CategoryTheory.pread …
    -/
  · ext; dsimp; apply preadditiveYoneda_homologySequenceδ_singleTriangle_apply
                /-
                  🎉 no goals
                -/
    /-
      case comm₂₃
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      S : CategoryTheory.ShortComplex C
      hS : S.ShortExact
      Y : C
      n₀ n₁ : Nat
      h : Eq (HAdd.hAdd 1 n₀) n₁
      this✝ : HasDerivedCategory C := HasDerivedCategory.standard C
      this : Function.Exact ⇑(CategoryTheory.ShortComplex.mk ((CategoryTheory.preadd …
      ⊢ Eq (AddMonoidHom.comp (CategoryTheory.ShortComplex.mk ((CategoryTheory.pread …
    -/
  · ext; apply singleFunctor_map_comp_hom (C := C)
         /-
           🎉 no goals
         -/


/-- Given a short exact short complex `S` in an abelian category `C` and an object `Y : C`,
this is the long exact sequence
`Ext S.X₃ Y n₀ → Ext S.X₂ Y n₀ → Ext S.X₁ Y n₀ → Ext S.X₃ Y n₁ → Ext S.X₂ Y n₁ → Ext S.X₁ Y n₁`
when `1 + n₀ = n₁`. -/
noncomputable def contravariantSequence : ComposableArrows AddCommGrp.{w} 5 :=
  mk₅ (AddCommGrp.ofHom ((mk₀ S.g).precomp Y (zero_add n₀)))
    (AddCommGrp.ofHom ((mk₀ S.f).precomp Y (zero_add n₀)))
    (AddCommGrp.ofHom (hS.extClass.precomp Y h))
    (AddCommGrp.ofHom ((mk₀ S.g).precomp Y (zero_add n₁)))
    (AddCommGrp.ofHom ((mk₀ S.f).precomp Y (zero_add n₁)))


lemma contravariantSequence_exact :
    (contravariantSequence hS Y n₀ n₁ h).Exact :=
  exact_of_δ₀ (contravariant_sequence_exact₂' hS Y n₀).exact_toComposableArrows
    (exact_of_δ₀ (contravariant_sequence_exact₁' hS Y n₀ n₁ h).exact_toComposableArrows
      (exact_of_δ₀ (contravariant_sequence_exact₃' hS Y n₀ n₁ h).exact_toComposableArrows
        (contravariant_sequence_exact₂' hS Y n₁).exact_toComposableArrows))


lemma contravariant_sequence_exact₁ {n₀ : ℕ} (x₁ : Ext S.X₁ Y n₀) {n₁ : ℕ} (hn₁ : 1 + n₀ = n₁)
    (hx₁ : hS.extClass.comp x₁ hn₁ = 0) :
    ∃ (x₂ : Ext S.X₂ Y n₀), (mk₀ S.f).comp x₂ (zero_add n₀) = x₁ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ : Nat
    x₁ : CategoryTheory.Abelian.Ext S.X₁ Y n₀
    n₁ : Nat
    hn₁ : Eq (HAdd.hAdd 1 n₀) n₁
    hx₁ : Eq (hS.extClass.comp x₁ hn₁) 0
    ⊢ Exists fun x₂ => Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x₂ ⋯) x₁
  -/
  have := contravariant_sequence_exact₁' hS Y n₀ n₁ hn₁
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ : Nat
    x₁ : CategoryTheory.Abelian.Ext S.X₁ Y n₀
    n₁ : Nat
    hn₁ : Eq (HAdd.hAdd 1 n₀) n₁
    hx₁ : Eq (hS.extClass.comp x₁ hn₁) 0
    this : (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abel …
    ⊢ Exists fun x₂ => Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x₂ ⋯) x₁
  -/
  rw [ShortComplex.ab_exact_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₀ : Nat
    x₁ : CategoryTheory.Abelian.Ext S.X₁ Y n₀
    n₁ : Nat
    hn₁ : Eq (HAdd.hAdd 1 n₀) n₁
    hx₁ : Eq (hS.extClass.comp x₁ hn₁) 0
    this : ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTh …
    ⊢ Exists fun x₂ => Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x₂ ⋯) x₁
  -/
  exact this x₁ hx₁
  /-
    🎉 no goals
  -/


include hS in
lemma contravariant_sequence_exact₂ {n : ℕ} (x₂ : Ext S.X₂ Y n)
    (hx₂ : (mk₀ S.f).comp x₂ (zero_add n) = 0) :
    ∃ (x₁ : Ext S.X₃ Y n), (mk₀ S.g).comp x₁ (zero_add n) = x₂ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n : Nat
    x₂ : CategoryTheory.Abelian.Ext S.X₂ Y n
    hx₂ : Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x₂ ⋯) 0
    ⊢ Exists fun x₁ => Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp x₁ ⋯) x₂
  -/
  have := contravariant_sequence_exact₂' hS Y n
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n : Nat
    x₂ : CategoryTheory.Abelian.Ext S.X₂ Y n
    hx₂ : Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x₂ ⋯) 0
    this : (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTheory.Abel …
    ⊢ Exists fun x₁ => Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp x₁ ⋯) x₂
  -/
  rw [ShortComplex.ab_exact_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n : Nat
    x₂ : CategoryTheory.Abelian.Ext S.X₂ Y n
    hx₂ : Eq ((CategoryTheory.Abelian.Ext.mk₀ S.f).comp x₂ ⋯) 0
    this : ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom ((CategoryTh …
    ⊢ Exists fun x₁ => Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp x₁ ⋯) x₂
  -/
  exact this x₂ hx₂
  /-
    🎉 no goals
  -/


lemma contravariant_sequence_exact₃ {n₁ : ℕ} (x₃ : Ext S.X₃ Y n₁)
    (hx₃ : (mk₀ S.g).comp x₃ (zero_add n₁) = 0) {n₀ : ℕ} (hn₀ : 1 + n₀ = n₁) :
    ∃ (x₁ : Ext S.X₁ Y n₀), hS.extClass.comp x₁ hn₀ = x₃ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₁ : Nat
    x₃ : CategoryTheory.Abelian.Ext S.X₃ Y n₁
    hx₃ : Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp x₃ ⋯) 0
    n₀ : Nat
    hn₀ : Eq (HAdd.hAdd 1 n₀) n₁
    ⊢ Exists fun x₁ => Eq (hS.extClass.comp x₁ hn₀) x₃
  -/
  have := contravariant_sequence_exact₃' hS Y n₀ n₁ hn₀
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₁ : Nat
    x₃ : CategoryTheory.Abelian.Ext S.X₃ Y n₁
    hx₃ : Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp x₃ ⋯) 0
    n₀ : Nat
    hn₀ : Eq (HAdd.hAdd 1 n₀) n₁
    this : (CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass.precomp  …
    ⊢ Exists fun x₁ => Eq (hS.extClass.comp x₁ hn₀) x₃
  -/
  rw [ShortComplex.ab_exact_iff] at this
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    S : CategoryTheory.ShortComplex C
    hS : S.ShortExact
    Y : C
    n₁ : Nat
    x₃ : CategoryTheory.Abelian.Ext S.X₃ Y n₁
    hx₃ : Eq ((CategoryTheory.Abelian.Ext.mk₀ S.g).comp x₃ ⋯) 0
    n₀ : Nat
    hn₀ : Eq (HAdd.hAdd 1 n₀) n₁
    this : ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (AddCommGrp.ofHom (hS.extClass …
    ⊢ Exists fun x₁ => Eq (hS.extClass.comp x₁ hn₀) x₃
  -/
  exact this x₃ hx₃
  /-
    🎉 no goals
  -/


