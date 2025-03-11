theorem ker_eq_bot_of_mono [Mono f] : LinearMap.ker f.hom = ⊥ :=
  LinearMap.ker_eq_bot_of_cancel fun u v h => ModuleCat.hom_ext_iff.mp <|
    (@cancel_mono _ _ _ _ _ f _ (↟u) (↟v)).1 <| ModuleCat.hom_ext_iff.mpr h


theorem range_eq_top_of_epi [Epi f] : LinearMap.range f.hom = ⊤ :=
  LinearMap.range_eq_top_of_cancel fun u v h => ModuleCat.hom_ext_iff.mp <|
    (@cancel_epi _ _ _ _ _ f _ (↟u) (↟v)).1 <| ModuleCat.hom_ext_iff.mpr h


theorem mono_iff_ker_eq_bot : Mono f ↔ LinearMap.ker f.hom = ⊥ :=
  ⟨fun _ => ker_eq_bot_of_mono _, fun hf =>
                                               /-
                                                 R : Type u
                                                 inst✝ : Ring R
                                                 X Y : ModuleCat R
                                                 f : Quiver.Hom X Y
                                                 hf : Eq (LinearMap.ker f.hom) Bot.bot
                                                 ⊢ Function.Injective ⇑f
                                               -/
    ConcreteCategory.mono_of_injective _ <| by convert LinearMap.ker_eq_bot.1 hf⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem mono_iff_injective : Mono f ↔ Function.Injective f := by
  /-
    R : Type u
    inst✝ : Ring R
    X Y : ModuleCat R
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Mono f) (Function.Injective ⇑f.hom)
  -/
  rw [mono_iff_ker_eq_bot, LinearMap.ker_eq_bot]
  /-
    🎉 no goals
  -/


theorem epi_iff_range_eq_top : Epi f ↔ LinearMap.range f.hom = ⊤ :=
  ⟨fun _ => range_eq_top_of_epi _, fun hf =>
                                               /-
                                                 R : Type u
                                                 inst✝ : Ring R
                                                 X Y : ModuleCat R
                                                 f : Quiver.Hom X Y
                                                 hf : Eq (LinearMap.range f.hom) Top.top
                                                 ⊢ Function.Surjective ⇑f
                                               -/
    ConcreteCategory.epi_of_surjective _ <| by convert LinearMap.range_eq_top.1 hf⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem epi_iff_surjective : Epi f ↔ Function.Surjective f := by
  /-
    R : Type u
    inst✝ : Ring R
    X Y : ModuleCat R
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑f.hom)
  -/
  rw [epi_iff_range_eq_top, LinearMap.range_eq_top]
  /-
    🎉 no goals
  -/


/-- If the zero morphism is an epi then the codomain is trivial. -/
def uniqueOfEpiZero (X) [h : Epi (0 : X ⟶ of R M)] : Unique M :=
  uniqueOfSurjectiveZero X ((ModuleCat.epi_iff_surjective _).mp h)


instance mono_as_hom'_subtype (U : Submodule R X) : Mono (ModuleCat.ofHom U.subtype) :=
  (mono_iff_ker_eq_bot _).mpr (Submodule.ker_subtype U)


instance epi_as_hom''_mkQ (U : Submodule R X) : Epi (ModuleCat.ofHom U.mkQ) :=
  (epi_iff_range_eq_top _).mpr <| Submodule.range_mkQ _


instance forget_preservesEpimorphisms : (forget (ModuleCat.{v} R)).PreservesEpimorphisms where
    preserves f hf := by
      /-
        R : Type u
        inst✝² : Ring R
        X Y : ModuleCat R
        f✝ : Quiver.Hom X Y
        M : Type v
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        X✝ Y✝ : ModuleCat R
        f : Quiver.Hom X✝ Y✝
        hf : CategoryTheory.Epi f
        ⊢ CategoryTheory.Epi ((CategoryTheory.forget (ModuleCat R)).map f)
      -/
      erw [CategoryTheory.epi_iff_surjective, ← epi_iff_surjective]
      /-
        R : Type u
        inst✝² : Ring R
        X Y : ModuleCat R
        f✝ : Quiver.Hom X Y
        M : Type v
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        X✝ Y✝ : ModuleCat R
        f : Quiver.Hom X✝ Y✝
        hf : CategoryTheory.Epi f
        ⊢ CategoryTheory.Epi f
      -/
      exact hf
      /-
        🎉 no goals
      -/


instance forget_preservesMonomorphisms : (forget (ModuleCat.{v} R)).PreservesMonomorphisms where
    preserves f hf := by
      /-
        R : Type u
        inst✝² : Ring R
        X Y : ModuleCat R
        f✝ : Quiver.Hom X Y
        M : Type v
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        X✝ Y✝ : ModuleCat R
        f : Quiver.Hom X✝ Y✝
        hf : CategoryTheory.Mono f
        ⊢ CategoryTheory.Mono ((CategoryTheory.forget (ModuleCat R)).map f)
      -/
      erw [CategoryTheory.mono_iff_injective, ← mono_iff_injective]
      /-
        R : Type u
        inst✝² : Ring R
        X Y : ModuleCat R
        f✝ : Quiver.Hom X Y
        M : Type v
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        X✝ Y✝ : ModuleCat R
        f : Quiver.Hom X✝ Y✝
        hf : CategoryTheory.Mono f
        ⊢ CategoryTheory.Mono f
      -/
      exact hf
      /-
        🎉 no goals
      -/


