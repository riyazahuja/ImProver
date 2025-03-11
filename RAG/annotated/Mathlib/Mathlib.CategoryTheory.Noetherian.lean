/-- A noetherian object is an object
which does not have infinite increasing sequences of subobjects.

See https://stacks.math.columbia.edu/tag/0FCG
-/
class NoetherianObject (X : C) : Prop where
  subobject_gt_wellFounded' : WellFounded ((· > ·) : Subobject X → Subobject X → Prop)


lemma NoetherianObject.subobject_gt_wellFounded (X : C) [NoetherianObject X] :
    WellFounded ((· > ·) : Subobject X → Subobject X → Prop) :=
  NoetherianObject.subobject_gt_wellFounded'


/-- An artinian object is an object
which does not have infinite decreasing sequences of subobjects.

See https://stacks.math.columbia.edu/tag/0FCF
-/
class ArtinianObject (X : C) : Prop where
  subobject_lt_wellFounded' : WellFounded ((· < ·) : Subobject X → Subobject X → Prop)


lemma ArtinianObject.subobject_lt_wellFounded (X : C) [ArtinianObject X] :
    WellFounded ((· < ·) : Subobject X → Subobject X → Prop) :=
  ArtinianObject.subobject_lt_wellFounded'


/-- A category is noetherian if it is essentially small and all objects are noetherian. -/
class Noetherian extends EssentiallySmall C : Prop where
  noetherianObject : ∀ X : C, NoetherianObject X


/-- A category is artinian if it is essentially small and all objects are artinian. -/
class Artinian extends EssentiallySmall C : Prop where
  artinianObject : ∀ X : C, ArtinianObject X


theorem exists_simple_subobject {X : C} [ArtinianObject X] (h : ¬IsZero X) :
    ∃ Y : Subobject X, Simple (Y : C) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : CategoryTheory.ArtinianObject X
    h : Not (CategoryTheory.Limits.IsZero X)
    ⊢ Exists fun Y => CategoryTheory.Simple (CategoryTheory.Subobject.underlying.o …
  -/
  haveI : Nontrivial (Subobject X) := nontrivial_of_not_isZero h
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : CategoryTheory.ArtinianObject X
    h : Not (CategoryTheory.Limits.IsZero X)
    this : Nontrivial (CategoryTheory.Subobject X)
    ⊢ Exists fun Y => CategoryTheory.Simple (CategoryTheory.Subobject.underlying.o …
  -/
  haveI := isAtomic_of_orderBot_wellFounded_lt (ArtinianObject.subobject_lt_wellFounded X)
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : CategoryTheory.ArtinianObject X
    h : Not (CategoryTheory.Limits.IsZero X)
    this✝ : Nontrivial (CategoryTheory.Subobject X)
    this : IsAtomic (CategoryTheory.Subobject X)
    ⊢ Exists fun Y => CategoryTheory.Simple (CategoryTheory.Subobject.underlying.o …
  -/
  obtain ⟨Y, s⟩ := (IsAtomic.eq_bot_or_exists_atom_le (⊤ : Subobject X)).resolve_left top_ne_bot
  /-
    case intro
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : CategoryTheory.ArtinianObject X
    h : Not (CategoryTheory.Limits.IsZero X)
    this✝ : Nontrivial (CategoryTheory.Subobject X)
    this : IsAtomic (CategoryTheory.Subobject X)
    Y : CategoryTheory.Subobject X
    s : And (IsAtom Y) (LE.le Y Top.top)
    ⊢ Exists fun Y => CategoryTheory.Simple (CategoryTheory.Subobject.underlying.o …
  -/
  exact ⟨Y, (subobject_simple_iff_isAtom _).mpr s.1⟩
  /-
    🎉 no goals
  -/


/-- Choose an arbitrary simple subobject of a non-zero artinian object. -/
noncomputable def simpleSubobject {X : C} [ArtinianObject X] (h : ¬IsZero X) : C :=
  (exists_simple_subobject h).choose


/-- The monomorphism from the arbitrary simple subobject of a non-zero artinian object. -/
noncomputable def simpleSubobjectArrow {X : C} [ArtinianObject X] (h : ¬IsZero X) :
    simpleSubobject h ⟶ X :=
  (exists_simple_subobject h).choose.arrow


instance mono_simpleSubobjectArrow {X : C} [ArtinianObject X] (h : ¬IsZero X) :
    Mono (simpleSubobjectArrow h) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : CategoryTheory.ArtinianObject X
    h : Not (CategoryTheory.Limits.IsZero X)
    ⊢ CategoryTheory.Mono (CategoryTheory.simpleSubobjectArrow h)
  -/
  dsimp only [simpleSubobjectArrow]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X : C
    inst✝ : CategoryTheory.ArtinianObject X
    h : Not (CategoryTheory.Limits.IsZero X)
    ⊢ CategoryTheory.Mono ⋯.choose.arrow
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance {X : C} [ArtinianObject X] (h : ¬IsZero X) : Simple (simpleSubobject h) :=
  (exists_simple_subobject h).choose_spec


