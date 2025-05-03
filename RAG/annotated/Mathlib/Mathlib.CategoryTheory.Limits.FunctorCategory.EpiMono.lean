instance [Mono f] (k : K) : Mono (f.app k) :=
  inferInstanceAs (Mono (((evaluation K C).obj k).map f))


lemma NatTrans.mono_iff_mono_app : Mono f ↔ ∀ (k : K), Mono (f.app k) :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ mono_of_mono_app _⟩


instance [Mono f] (H : C ⥤ D) [H.PreservesMonomorphisms] :
    Mono (whiskerRight f H) := by
  /-
    K : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} K
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    F G : CategoryTheory.Functor K C
    f : Quiver.Hom F G
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    inst✝¹ : CategoryTheory.Mono f
    H : CategoryTheory.Functor C D
    inst✝ : H.PreservesMonomorphisms
    ⊢ CategoryTheory.Mono (CategoryTheory.whiskerRight f H)
  -/
  have : ∀ X, Mono ((whiskerRight f H).app X) := by intros; dsimp; infer_instance
  /-
    K : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} K
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    F G : CategoryTheory.Functor K C
    f : Quiver.Hom F G
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    inst✝¹ : CategoryTheory.Mono f
    H : CategoryTheory.Functor C D
    inst✝ : H.PreservesMonomorphisms
    this : ∀ (X : K), CategoryTheory.Mono ((CategoryTheory.whiskerRight f H).app X)
    ⊢ CategoryTheory.Mono (CategoryTheory.whiskerRight f H)
  -/
  apply NatTrans.mono_of_mono_app
  /-
    🎉 no goals
  -/


instance [Epi f] (k : K) : Epi (f.app k) :=
  inferInstanceAs (Epi (((evaluation K C).obj k).map f))


lemma NatTrans.epi_iff_epi_app : Epi f ↔ ∀ (k : K), Epi (f.app k) :=
  ⟨fun _ ↦ inferInstance, fun _ ↦ epi_of_epi_app _⟩


instance [Epi f] (H : C ⥤ D) [H.PreservesEpimorphisms] :
    Epi (whiskerRight f H) := by
  /-
    K : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} K
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    F G : CategoryTheory.Functor K C
    f : Quiver.Hom F G
    inst✝² : CategoryTheory.Limits.HasPushouts C
    inst✝¹ : CategoryTheory.Epi f
    H : CategoryTheory.Functor C D
    inst✝ : H.PreservesEpimorphisms
    ⊢ CategoryTheory.Epi (CategoryTheory.whiskerRight f H)
  -/
  have : ∀ X, Epi ((whiskerRight f H).app X) := by intros; dsimp; infer_instance
  /-
    K : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} K
    C : Type u'
    inst✝⁴ : CategoryTheory.Category.{v', u'} C
    D : Type u''
    inst✝³ : CategoryTheory.Category.{v'', u''} D
    F G : CategoryTheory.Functor K C
    f : Quiver.Hom F G
    inst✝² : CategoryTheory.Limits.HasPushouts C
    inst✝¹ : CategoryTheory.Epi f
    H : CategoryTheory.Functor C D
    inst✝ : H.PreservesEpimorphisms
    this : ∀ (X : K), CategoryTheory.Epi ((CategoryTheory.whiskerRight f H).app X)
    ⊢ CategoryTheory.Epi (CategoryTheory.whiskerRight f H)
  -/
  apply NatTrans.epi_of_epi_app
  /-
    🎉 no goals
  -/


