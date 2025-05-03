instance [Precoherent C] [HasFiniteCoproducts C] : Preregular C where
  exists_fac {X Y Z} f g _ := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      ⊢ Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Category …
    -/
    have hp := Precoherent.pullback f PUnit (fun () ↦ Z) (fun () ↦ g)
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      hp : (CategoryTheory.EffectiveEpiFamily (fun x => Z) fun x => CategoryTheory.i …
      ⊢ Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Category …
    -/
    simp only [exists_const] at hp
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      hp : (CategoryTheory.EffectiveEpiFamily (fun x => Z) fun x => g) → Exists fun  …
      ⊢ Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Category …
    -/
    rw [← effectiveEpi_iff_effectiveEpiFamily g] at hp
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      hp : CategoryTheory.EffectiveEpi g → Exists fun β => Exists fun h => Exists fu …
      ⊢ Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Category …
    -/
    obtain ⟨β, _, X₂, π₂, h, ι, hι⟩ := hp inferInstance
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      hp : CategoryTheory.EffectiveEpi g → Exists fun β => Exists fun h => Exists fu …
      β : Type
      w✝ : Finite β
      X₂ : β → C
      π₂ : (b : β) → Quiver.Hom (X₂ b) X
      h : CategoryTheory.EffectiveEpiFamily X₂ π₂
      ι : (b : β) → Quiver.Hom (X₂ b) Z
      hι : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) g) (CategoryTheor …
      ⊢ Exists fun W => Exists fun h => Exists fun x => Exists fun i => Eq (Category …
    -/
    refine ⟨∐ X₂, Sigma.desc π₂, inferInstance, Sigma.desc ι, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      hp : CategoryTheory.EffectiveEpi g → Exists fun β => Exists fun h => Exists fu …
      β : Type
      w✝ : Finite β
      X₂ : β → C
      π₂ : (b : β) → Quiver.Hom (X₂ b) X
      h : CategoryTheory.EffectiveEpiFamily X₂ π₂
      ι : (b : β) → Quiver.Hom (X₂ b) Z
      hι : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) g) (CategoryTheor …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.desc ι)  …
    -/
    ext b
    /-
      case intro.intro.intro.intro.intro.intro.h
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.Precoherent C
      inst✝ : CategoryTheory.Limits.HasFiniteCoproducts C
      X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom Z Y
      x✝ : CategoryTheory.EffectiveEpi g
      hp : CategoryTheory.EffectiveEpi g → Exists fun β => Exists fun h => Exists fu …
      β : Type
      w✝ : Finite β
      X₂ : β → C
      π₂ : (b : β) → Quiver.Hom (X₂ b) X
      h : CategoryTheory.EffectiveEpiFamily X₂ π₂
      ι : (b : β) → Quiver.Hom (X₂ b) Z
      hι : ∀ (b : β), Eq (CategoryTheory.CategoryStruct.comp (ι b) g) (CategoryTheor …
      b : β
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι X₂ b)  …
    -/
    simpa using hι b
    /-
      🎉 no goals
    -/


instance [FinitaryPreExtensive C] [Preregular C] : Precoherent C where
  pullback {B₁ B₂} f α _ X₁ π₁ h := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      ⊢ Exists fun β => Exists fun x => Exists fun X₂ => Exists fun π₂ => And (Categ …
    -/
    refine ⟨α, inferInstance, ?_⟩
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      ⊢ Exists fun X₂ => Exists fun π₂ => And (CategoryTheory.EffectiveEpiFamily X₂  …
    -/
    obtain ⟨Y, g, _, g', hg⟩ := Preregular.exists_fac f (Sigma.desc π₁)
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      Y : C
      g : Quiver.Hom Y B₂
      w✝ : CategoryTheory.EffectiveEpi g
      g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
      hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
      ⊢ Exists fun X₂ => Exists fun π₂ => And (CategoryTheory.EffectiveEpiFamily X₂  …
    -/
    let X₂ := fun a ↦ pullback g' (Sigma.ι X₁ a)
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      Y : C
      g : Quiver.Hom Y B₂
      w✝ : CategoryTheory.EffectiveEpi g
      g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
      hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
      X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
      ⊢ Exists fun X₂ => Exists fun π₂ => And (CategoryTheory.EffectiveEpiFamily X₂  …
    -/
    let π₂ := fun a ↦ pullback.fst g' (Sigma.ι X₁ a) ≫ g
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      Y : C
      g : Quiver.Hom Y B₂
      w✝ : CategoryTheory.EffectiveEpi g
      g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
      hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
      X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
      π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
      ⊢ Exists fun X₂ => Exists fun π₂ => And (CategoryTheory.EffectiveEpiFamily X₂  …
    -/
    let π' := fun a ↦ pullback.fst g' (Sigma.ι X₁ a)
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      Y : C
      g : Quiver.Hom Y B₂
      w✝ : CategoryTheory.EffectiveEpi g
      g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
      hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
      X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
      π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
      π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
      ⊢ Exists fun X₂ => Exists fun π₂ => And (CategoryTheory.EffectiveEpiFamily X₂  …
    -/
    have _ := FinitaryPreExtensive.sigma_desc_iso (fun a ↦ Sigma.ι X₁ a) g' inferInstance
    /-
      case intro.intro.intro.intro
      C : Type u_1
      inst✝² : CategoryTheory.Category.{u_2, u_1} C
      inst✝¹ : CategoryTheory.FinitaryPreExtensive C
      inst✝ : CategoryTheory.Preregular C
      B₁ B₂ : C
      f : Quiver.Hom B₂ B₁
      α : Type
      x✝¹ : Finite α
      X₁ : α → C
      π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
      h : CategoryTheory.EffectiveEpiFamily X₁ π₁
      Y : C
      g : Quiver.Hom Y B₂
      w✝ : CategoryTheory.EffectiveEpi g
      g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
      hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
      X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
      π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
      π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
      x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
      ⊢ Exists fun X₂ => Exists fun π₂ => And (CategoryTheory.EffectiveEpiFamily X₂  …
    -/
    refine ⟨X₂, π₂, ?_, ?_⟩
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        ⊢ CategoryTheory.EffectiveEpiFamily X₂ π₂
      -/
    · have : (Sigma.desc π' ≫ g) = Sigma.desc π₂ := by ext; simp [π₂, π']
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.des …
        ⊢ CategoryTheory.EffectiveEpiFamily X₂ π₂
      -/
      rw [← effectiveEpi_desc_iff_effectiveEpiFamily, ← this]
      /-
        case intro.intro.intro.intro.refine_1
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.des …
        ⊢ CategoryTheory.EffectiveEpi (CategoryTheory.CategoryStruct.comp (CategoryThe …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        ⊢ Exists fun i => Exists fun ι => ∀ (b : α), Eq (CategoryTheory.CategoryStruct …
      -/
    · refine ⟨id, fun b ↦ pullback.snd _ _, fun b ↦ ?_⟩
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        b : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun b => CategoryTheory.Limits.pull …
      -/
      simp only [X₂, π₂, id_eq, Category.assoc, ← hg]
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        b : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd g …
      -/
      rw [← Category.assoc, pullback.condition]
      /-
        case intro.intro.intro.intro.refine_2
        C : Type u_1
        inst✝² : CategoryTheory.Category.{u_2, u_1} C
        inst✝¹ : CategoryTheory.FinitaryPreExtensive C
        inst✝ : CategoryTheory.Preregular C
        B₁ B₂ : C
        f : Quiver.Hom B₂ B₁
        α : Type
        x✝¹ : Finite α
        X₁ : α → C
        π₁ : (a : α) → Quiver.Hom (X₁ a) B₁
        h : CategoryTheory.EffectiveEpiFamily X₁ π₁
        Y : C
        g : Quiver.Hom Y B₂
        w✝ : CategoryTheory.EffectiveEpi g
        g' : Quiver.Hom Y (CategoryTheory.Limits.sigmaObj X₁)
        hg : Eq (CategoryTheory.CategoryStruct.comp g' (CategoryTheory.Limits.Sigma.de …
        X₂ : α → C := fun a => CategoryTheory.Limits.pullback g' (CategoryTheory.Limit …
        π₂ : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        π' : (a : α) → Quiver.Hom (CategoryTheory.Limits.pullback g' (CategoryTheory.L …
        x✝ : CategoryTheory.IsIso (CategoryTheory.Limits.Sigma.desc fun x => CategoryT …
        b : α
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd g …
      -/
      simp
      /-
        🎉 no goals
      -/


/-- The union of the extensive and regular coverages generates the coherent topology on `C`. -/
theorem extensive_regular_generate_coherent [Preregular C] [FinitaryPreExtensive C] :
    ((extensiveCoverage C) ⊔ (regularCoverage C)).toGrothendieck =
    (coherentTopology C) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    ⊢ Eq (CategoryTheory.Coverage.toGrothendieck C (Max.max (CategoryTheory.extens …
  -/
  ext B S
  /-
    case h.h.h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preregular C
    inst✝ : CategoryTheory.FinitaryPreExtensive C
    B : C
    S : CategoryTheory.Sieve B
    ⊢ Iff (Membership.mem ((CategoryTheory.Coverage.toGrothendieck C (Max.max (Cat …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
  · induction h with
    | of Y T hT =>
      apply Coverage.Saturate.of
      simp only [Coverage.sup_covering, Set.mem_union] at hT
      exact Or.elim hT
        (fun ⟨α, x, X, π, ⟨h, _⟩⟩ ↦ ⟨α, x, X, π, ⟨h, inferInstance⟩⟩)
        (fun ⟨Z, f, ⟨h, _⟩⟩ ↦ ⟨Unit, inferInstance, fun _ ↦ Z, fun _ ↦ f, ⟨h, inferInstance⟩⟩)
    | top => apply Coverage.Saturate.top
    | transitive Y T => apply Coverage.Saturate.transitive Y T<;> [assumption; assumption]
  · induction h with
    | of Y T hT =>
      obtain ⟨I, _, X, f, rfl, hT⟩ := hT
      apply Coverage.Saturate.transitive Y (generate (Presieve.ofArrows
        (fun (_ : Unit) ↦ (∐ fun (i : I) => X i)) (fun (_ : Unit) ↦ Sigma.desc f)))
      · apply Coverage.Saturate.of
        simp only [Coverage.sup_covering, extensiveCoverage, regularCoverage, Set.mem_union,
          Set.mem_setOf_eq]
        exact Or.inr ⟨_, Sigma.desc f, ⟨rfl, inferInstance⟩⟩
      · rintro R g ⟨W, ψ, σ, ⟨⟩, rfl⟩
        change _ ∈ ((extensiveCoverage C) ⊔ (regularCoverage C)).toGrothendieck _ R
        rw [Sieve.pullback_comp]
        apply pullback_stable
        have : generate (Presieve.ofArrows X fun (i : I) ↦ Sigma.ι X i) ≤
            (generate (Presieve.ofArrows X f)).pullback (Sigma.desc f) := by
          rintro Q q ⟨E, e, r, ⟨hq, rfl⟩⟩
          exact ⟨E, e, r ≫ (Sigma.desc f), by cases hq; simpa using Presieve.ofArrows.mk _, by simp⟩
        apply Coverage.saturate_of_superset _ this
        apply Coverage.Saturate.of
        refine Or.inl ⟨I, inferInstance, _, _, ⟨rfl, ?_⟩⟩
        convert IsIso.id _
        aesop
    | top => apply Coverage.Saturate.top
    | transitive Y T => apply Coverage.Saturate.transitive Y T<;> [assumption; assumption]


