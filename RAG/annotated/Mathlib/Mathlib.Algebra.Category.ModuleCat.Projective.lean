/-- The categorical notion of projective object agrees with the explicit module-theoretic notion. -/
theorem IsProjective.iff_projective {R : Type u} [Ring R] {P : Type max u v} [AddCommGroup P]
    [Module R P] : Module.Projective R P ↔ Projective (ModuleCat.of R P) := by
  /-
    R : Type u
    inst✝² : Ring R
    P : Type (max u v)
    inst✝¹ : AddCommGroup P
    inst✝ : Module R P
    ⊢ Iff (Module.Projective R P) (CategoryTheory.Projective (ModuleCat.of R P))
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h : Module.Projective R P
      ⊢ CategoryTheory.Projective (ModuleCat.of R P)
    -/
  · letI : Module.Projective R (ModuleCat.of R P) := h
    /-
      case refine_1
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h : Module.Projective R P
      this : Module.Projective R ↑(ModuleCat.of R P) := h
      ⊢ CategoryTheory.Projective (ModuleCat.of R P)
    -/
    refine ⟨fun E X epi => ?_⟩
    obtain ⟨f, h⟩ := Module.projective_lifting_property X.hom E.hom
      ((ModuleCat.epi_iff_surjective _).mp epi)
    /-
      case refine_1.intro
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h✝ : Module.Projective R P
      this : Module.Projective R ↑(ModuleCat.of R P) := h✝
      E✝ X✝ : ModuleCat R
      E : Quiver.Hom (ModuleCat.of R P) X✝
      X : Quiver.Hom E✝ X✝
      epi : CategoryTheory.Epi X
      f : LinearMap (RingHom.id R) ↑(ModuleCat.of R P) ↑E✝
      h : Eq (X.hom.comp f) E.hom
      ⊢ Exists fun f' => Eq (CategoryTheory.CategoryStruct.comp f' X) E
    -/
    exact ⟨ofHom f, hom_ext h⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h : CategoryTheory.Projective (ModuleCat.of R P)
      ⊢ Module.Projective R P
    -/
  · refine Module.Projective.of_lifting_property.{u,v} ?_
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h : CategoryTheory.Projective (ModuleCat.of R P)
      ⊢ ∀ {M : Type (max v u)} {N : Type (max u v)} [inst : AddCommGroup M] [inst_1  …
    -/
    intro E X mE mX sE sX f g s
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h : CategoryTheory.Projective (ModuleCat.of R P)
      E : Type (max v u)
      X : Type (max u v)
      mE : AddCommGroup E
      mX : AddCommGroup X
      sE : Module R E
      sX : Module R X
      f : LinearMap (RingHom.id R) E X
      g : LinearMap (RingHom.id R) P X
      s : Function.Surjective ⇑f
      ⊢ Exists fun h => Eq (f.comp h) g
    -/
    haveI : Epi (↟f) := (ModuleCat.epi_iff_surjective (↟f)).mpr s
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      P : Type (max u v)
      inst✝¹ : AddCommGroup P
      inst✝ : Module R P
      h : CategoryTheory.Projective (ModuleCat.of R P)
      E : Type (max v u)
      X : Type (max u v)
      mE : AddCommGroup E
      mX : AddCommGroup X
      sE : Module R E
      sX : Module R X
      f : LinearMap (RingHom.id R) E X
      g : LinearMap (RingHom.id R) P X
      s : Function.Surjective ⇑f
      this : CategoryTheory.Epi (ModuleCat.ofHom f)
      ⊢ Exists fun h => Eq (f.comp h) g
    -/
    letI : Projective (ModuleCat.of R P) := h
    exact ⟨(Projective.factorThru (↟g) (↟f)).hom,
      ModuleCat.hom_ext_iff.mp <| Projective.factorThru_comp (↟g) (↟f)⟩


/-- Modules that have a basis are projective. -/
theorem projective_of_free {ι : Type u'} (b : Basis ι R M) : Projective M :=
  Projective.of_iso (ModuleCat.ofSelfIso M)
    (IsProjective.iff_projective.{v,u}.mp (Module.Projective.of_basis b))


/-- The category of modules has enough projectives, since every module is a quotient of a free
    module. -/
instance moduleCat_enoughProjectives : EnoughProjectives (ModuleCat.{max u v} R) where
  presentation M :=
    ⟨{  p := ModuleCat.of R (M →₀ R)
        projective :=
          projective_of_free.{v,u} (ι := M) (M := ModuleCat.of R (M →₀ R)) <|
            Finsupp.basisSingleOne
        f := ofHom <| Finsupp.basisSingleOne.constr ℕ _root_.id
        epi := (epi_iff_range_eq_top _).mpr
            (range_eq_top.2 fun m => ⟨Finsupp.single m (1 : R), by
              /-
                R : Type u
                inst✝ : Ring R
                M✝ M : ModuleCat R
                m : ↑M
                ⊢ Eq ((ModuleCat.ofHom ((Finsupp.basisSingleOne.constr Nat) id)).hom (Finsupp. …
              -/
              simp [Finsupp.linearCombination_single, Basis.constr] ⟩)}⟩
              /-
                🎉 no goals
              -/


