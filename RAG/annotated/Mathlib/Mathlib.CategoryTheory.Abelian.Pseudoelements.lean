/-- This is just composition of morphisms in `C`. Another way to express this would be
    `(Over.map f).obj a`, but our definition has nicer definitional properties. -/
def app {P Q : C} (f : P ⟶ Q) (a : Over P) : Over Q :=
  a.hom ≫ f


@[simp]
theorem app_hom {P Q : C} (f : P ⟶ Q) (a : Over P) : (app f a).hom = a.hom ≫ f := rfl


/-- Two arrows `f : X ⟶ P` and `g : Y ⟶ P` are called pseudo-equal if there is some object
    `R` and epimorphisms `p : R ⟶ X` and `q : R ⟶ Y` such that `p ≫ f = q ≫ g`. -/
def PseudoEqual (P : C) (f g : Over P) : Prop :=
  ∃ (R : C) (p : R ⟶ f.1) (q : R ⟶ g.1) (_ : Epi p) (_ : Epi q), p ≫ f.hom = q ≫ g.hom


theorem pseudoEqual_refl {P : C} : Reflexive (PseudoEqual P) :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝ : CategoryTheory.Category.{v, u} C
                                                                  P : C
                                                                  f : CategoryTheory.Over P
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id f.l …
                                                                -/
  fun f => ⟨f.1, 𝟙 f.1, 𝟙 f.1, inferInstance, inferInstance, by simp⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem pseudoEqual_symm {P : C} : Symmetric (PseudoEqual P) :=
  fun _ _ ⟨R, p, q, ep, Eq, comm⟩ => ⟨R, q, p, Eq, ep, comm.symm⟩


/-- Pseudoequality is transitive: Just take the pullback. The pullback morphisms will
    be epimorphisms since in an abelian category, pullbacks of epimorphisms are epimorphisms. -/
theorem pseudoEqual_trans {P : C} : Transitive (PseudoEqual P) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    ⊢ Transitive (CategoryTheory.Abelian.PseudoEqual P)
  -/
  intro f g h ⟨R, p, q, ep, Eq, comm⟩ ⟨R', p', q', ep', eq', comm'⟩
  refine ⟨pullback q p', pullback.fst _ _ ≫ p, pullback.snd _ _ ≫ q',
    epi_comp _ _, epi_comp _ _, ?_⟩
  rw [Category.assoc, comm, ← Category.assoc, pullback.condition, Category.assoc, comm',
    Category.assoc]


/-- The arrows with codomain `P` equipped with the equivalence relation of being pseudo-equal. -/
def Pseudoelement.setoid (P : C) : Setoid (Over P) :=
  ⟨_, ⟨pseudoEqual_refl, @pseudoEqual_symm _ _ _, @pseudoEqual_trans _ _ _ _⟩⟩


/-- A `Pseudoelement` of `P` is just an equivalence class of arrows ending in `P` by being
    pseudo-equal. -/
def Pseudoelement (P : C) : Type max u v :=
  Quotient (Pseudoelement.setoid P)


/-- A coercion from an object of an abelian category to its pseudoelements. -/
def objectToSort : CoeSort C (Type max u v) :=
  ⟨fun P => Pseudoelement P⟩


/-- A coercion from an arrow with codomain `P` to its associated pseudoelement. -/
def overToSort {P : C} : Coe (Over P) (Pseudoelement P) :=
  ⟨Quot.mk (PseudoEqual P)⟩


theorem over_coe_def {P Q : C} (a : Q ⟶ P) : (a : Pseudoelement P) = ⟦↑a⟧ := rfl


/-- If two elements are pseudo-equal, then their composition with a morphism is, too. -/
theorem pseudoApply_aux {P Q : C} (f : P ⟶ Q) (a b : Over P) : a ≈ b → app f a ≈ app f b :=
  fun ⟨R, p, q, ep, Eq, comm⟩ =>
                                                          /-
                                                            C : Type u
                                                            inst✝¹ : CategoryTheory.Category.{v, u} C
                                                            inst✝ : CategoryTheory.Abelian C
                                                            P Q : C
                                                            f : Quiver.Hom P Q
                                                            a b : CategoryTheory.Over P
                                                            x✝ : HasEquiv.Equiv a b
                                                            R : C
                                                            p : Quiver.Hom R a.left
                                                            q : Quiver.Hom R b.left
                                                            ep : CategoryTheory.Epi p
                                                            Eq : CategoryTheory.Epi q
                                                            comm : _root_.Eq (CategoryTheory.CategoryStruct.comp p a.hom) (CategoryTheory. …
                                                            ⊢ _root_.Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.CategoryStru …
                                                          -/
  ⟨R, p, q, ep, Eq, show p ≫ a.hom ≫ f = q ≫ b.hom ≫ f by rw [reassoc_of% comm]⟩
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- A morphism `f` induces a function `pseudoApply f` on pseudoelements. -/
def pseudoApply {P Q : C} (f : P ⟶ Q) : P → Q :=
  Quotient.map (fun g : Over P => app f g) (pseudoApply_aux f)


/-- A coercion from morphisms to functions on pseudoelements. -/
def homToFun {P Q : C} : CoeFun (P ⟶ Q) fun _ => P → Q :=
  ⟨pseudoApply⟩


theorem pseudoApply_mk' {P Q : C} (f : P ⟶ Q) (a : Over P) : f ⟦a⟧ = ⟦↑(a.hom ≫ f)⟧ := rfl


/-- Applying a pseudoelement to a composition of morphisms is the same as composing
    with each morphism. Sadly, this is not a definitional equality, but at least it is
    true. -/
theorem comp_apply {P Q R : C} (f : P ⟶ Q) (g : Q ⟶ R) (a : P) : (f ≫ g) a = g (f a) :=
  Quotient.inductionOn a fun x =>
    Quotient.sound <| by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        a : CategoryTheory.Abelian.Pseudoelement P
        x : CategoryTheory.Over P
        ⊢ HasEquiv.Equiv ((fun g_1 => CategoryTheory.Abelian.app (CategoryTheory.Categ …
      -/
      simp only [app]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        P Q R : C
        f : Quiver.Hom P Q
        g : Quiver.Hom Q R
        a : CategoryTheory.Abelian.Pseudoelement P
        x : CategoryTheory.Over P
        ⊢ HasEquiv.Equiv (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp x …
      -/
      rw [← Category.assoc, Over.coe_hom]
      /-
        🎉 no goals
      -/


/-- Composition of functions on pseudoelements is composition of morphisms. -/
theorem comp_comp {P Q R : C} (f : P ⟶ Q) (g : Q ⟶ R) : g ∘ f = f ≫ g :=
  funext fun _ => (comp_apply _ _ _).symm


/-- The arrows pseudo-equal to a zero morphism are precisely the zero morphisms. -/
theorem pseudoZero_aux {P : C} (Q : C) (f : Over P) : f ≈ (0 : Q ⟶ P) ↔ f.hom = 0 :=
                                                       /-
                                                         C : Type u
                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                         inst✝ : CategoryTheory.Abelian C
                                                         P Q : C
                                                         f : CategoryTheory.Over P
                                                         x✝ : HasEquiv.Equiv f (CategoryTheory.Over.mk 0)
                                                         R : C
                                                         p : Quiver.Hom R f.left
                                                         q : Quiver.Hom R (CategoryTheory.Over.mk 0).left
                                                         w✝¹ : CategoryTheory.Epi p
                                                         w✝ : CategoryTheory.Epi q
                                                         comm : Eq (CategoryTheory.CategoryStruct.comp p f.hom) (CategoryTheory.Categor …
                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp p f.hom) 0
                                                       -/
  ⟨fun ⟨R, p, q, _, _, comm⟩ => zero_of_epi_comp p (by simp [comm]), fun hf =>
                                                       /-
                                                         🎉 no goals
                                                       -/
    ⟨biprod f.1 Q, biprod.fst, biprod.snd, inferInstance, inferInstance, by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        P Q : C
        f : CategoryTheory.Over P
        hf : Eq f.hom 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp CategoryTheory.Limits.biprod.fst f.ho …
      -/
      rw [hf, Over.coe_hom, HasZeroMorphisms.comp_zero, HasZeroMorphisms.comp_zero]⟩⟩
      /-
        🎉 no goals
      -/


theorem zero_eq_zero' {P Q R : C} :
    (⟦((0 : Q ⟶ P) : Over P)⟧ : Pseudoelement P) = ⟦((0 : R ⟶ P) : Over P)⟧ :=
  Quotient.sound <| (pseudoZero_aux R _).2 rfl


/-- The zero pseudoelement is the class of a zero morphism. -/
def pseudoZero {P : C} : P :=
  ⟦(0 : P ⟶ P)⟧

-- Porting note: in mathlib3, we couldn't make this an instance
-- as it would have fired on `coe_sort`.
-- However now that coercions are treated differently, this is a structural instance triggered by
-- the appearance of `Pseudoelement`.

instance hasZero {P : C} : Zero P :=
  ⟨pseudoZero⟩


instance {P : C} : Inhabited P :=
  ⟨0⟩


theorem pseudoZero_def {P : C} : (0 : Pseudoelement P) = ⟦↑(0 : P ⟶ P)⟧ := rfl


@[simp]
theorem zero_eq_zero {P Q : C} : ⟦((0 : Q ⟶ P) : Over P)⟧ = (0 : Pseudoelement P) :=
  zero_eq_zero'


/-- The pseudoelement induced by an arrow is zero precisely when that arrow is zero. -/
theorem pseudoZero_iff {P : C} (a : Over P) : a = (0 : P) ↔ a.hom = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    a : CategoryTheory.Over P
    ⊢ Iff (Eq (Quot.mk (CategoryTheory.Abelian.PseudoEqual P) a) 0) (Eq a.hom 0)
  -/
  rw [← pseudoZero_aux P a]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P : C
    a : CategoryTheory.Over P
    ⊢ Iff (Eq (Quot.mk (CategoryTheory.Abelian.PseudoEqual P) a) 0) (HasEquiv.Equi …
  -/
  exact Quotient.eq'
  /-
    🎉 no goals
  -/


/-- Morphisms map the zero pseudoelement to the zero pseudoelement. -/
@[simp]
theorem apply_zero {P Q : C} (f : P ⟶ Q) : f 0 = 0 := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f 0) 0
  -/
  rw [pseudoZero_def, pseudoApply_mk']
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (CategoryThe …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The zero morphism maps every pseudoelement to 0. -/
@[simp]
theorem zero_apply {P : C} (Q : C) (a : P) : (0 : P ⟶ Q) a = 0 :=
  Quotient.inductionOn a fun a' => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      P Q : C
      a : CategoryTheory.Abelian.Pseudoelement P
      a' : CategoryTheory.Over P
      ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply 0 (Quotient.mk (Categor …
    -/
    rw [pseudoZero_def, pseudoApply_mk']
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.Abelian C
      P Q : C
      a : CategoryTheory.Abelian.Pseudoelement P
      a' : CategoryTheory.Over P
      ⊢ Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (CategoryThe …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- An extensionality lemma for being the zero arrow. -/
theorem zero_morphism_ext {P Q : C} (f : P ⟶ Q) : (∀ a, f a = 0) → f = 0 := fun h => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : ∀ (a : CategoryTheory.Abelian.Pseudoelement P), Eq (CategoryTheory.Abelian …
    ⊢ Eq f 0
  -/
  rw [← Category.id_comp f]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : ∀ (a : CategoryTheory.Abelian.Pseudoelement P), Eq (CategoryTheory.Abelian …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id P)  …
  -/
  exact (pseudoZero_iff (𝟙 P ≫ f : Over Q)).1 (h (𝟙 P))
  /-
    🎉 no goals
  -/


theorem zero_morphism_ext' {P Q : C} (f : P ⟶ Q) : (∀ a, f a = 0) → 0 = f :=
  Eq.symm ∘ zero_morphism_ext f


theorem eq_zero_iff {P Q : C} (f : P ⟶ Q) : f = 0 ↔ ∀ a, f a = 0 :=
                 /-
                   C : Type u
                   inst✝¹ : CategoryTheory.Category.{v, u} C
                   inst✝ : CategoryTheory.Abelian C
                   P Q : C
                   f : Quiver.Hom P Q
                   h : Eq f 0
                   a : CategoryTheory.Abelian.Pseudoelement P
                   ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f a) 0
                 -/
  ⟨fun h a => by simp [h], zero_morphism_ext _⟩
                 /-
                   🎉 no goals
                 -/


/-- A monomorphism is injective on pseudoelements. -/
theorem pseudo_injective_of_mono {P Q : C} (f : P ⟶ Q) [Mono f] : Function.Injective f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    inst✝ : CategoryTheory.Mono f
    ⊢ Function.Injective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
  -/
  intro abar abar'
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    inst✝ : CategoryTheory.Mono f
    abar abar' : CategoryTheory.Abelian.Pseudoelement P
    ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f abar) (CategoryTheory …
  -/
  refine Quotient.inductionOn₂ abar abar' fun a a' ha => ?_
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    inst✝ : CategoryTheory.Mono f
    abar abar' : CategoryTheory.Abelian.Pseudoelement P
    a a' : CategoryTheory.Over P
    ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Cate …
    ⊢ Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid P) a) (Quotient …
  -/
  apply Quotient.sound
  /-
    case a
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    inst✝ : CategoryTheory.Mono f
    abar abar' : CategoryTheory.Abelian.Pseudoelement P
    a a' : CategoryTheory.Over P
    ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Cate …
    ⊢ HasEquiv.Equiv a a'
  -/
  have : (⟦(a.hom ≫ f : Over Q)⟧ : Quotient (setoid Q)) = ⟦↑(a'.hom ≫ f)⟧ := by convert ha
  /-
    case a
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    inst✝ : CategoryTheory.Mono f
    abar abar' : CategoryTheory.Abelian.Pseudoelement P
    a a' : CategoryTheory.Over P
    ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Cate …
    this : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (Catego …
    ⊢ HasEquiv.Equiv a a'
  -/
  have ⟨R, p, q, ep, Eq, comm⟩ := Quotient.exact this
  exact ⟨R, p, q, ep, Eq, (cancel_mono f).1 <| by
    simp only [Category.assoc]
    exact comm⟩


/-- A morphism that is injective on pseudoelements only maps the zero element to zero. -/
theorem zero_of_map_zero {P Q : C} (f : P ⟶ Q) : Function.Injective f → ∀ a, f a = 0 → a = 0 :=
  fun h a ha => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Injective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    a : CategoryTheory.Abelian.Pseudoelement P
    ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f a) 0
    ⊢ Eq a 0
  -/
  rw [← apply_zero f] at ha
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Injective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    a : CategoryTheory.Abelian.Pseudoelement P
    ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f a) (CategoryTheory …
    ⊢ Eq a 0
  -/
  exact h ha
  /-
    🎉 no goals
  -/


/-- A morphism that only maps the zero pseudoelement to zero is a monomorphism. -/
theorem mono_of_zero_of_map_zero {P Q : C} (f : P ⟶ Q) : (∀ a, f a = 0 → a = 0) → Mono f :=
  fun h => (mono_iff_cancel_zero _).2 fun _ g hg =>
    (pseudoZero_iff (g : Over P)).1 <|
      h _ <| show f g = 0 from (pseudoZero_iff (g ≫ f : Over Q)).2 hg


/-- An epimorphism is surjective on pseudoelements. -/
theorem pseudo_surjective_of_epi {P Q : C} (f : P ⟶ Q) [Epi f] : Function.Surjective f :=
  fun qbar =>
  Quotient.inductionOn qbar fun q =>
    ⟨(pullback.fst f q.hom : Over P),
      Quotient.sound <|
        ⟨pullback f q.hom, 𝟙 (pullback f q.hom), pullback.snd _ _, inferInstance, inferInstance, by
          /-
            C : Type u
            inst✝² : CategoryTheory.Category.{v, u} C
            inst✝¹ : CategoryTheory.Abelian C
            P Q : C
            f : Quiver.Hom P Q
            inst✝ : CategoryTheory.Epi f
            qbar : CategoryTheory.Abelian.Pseudoelement Q
            q : CategoryTheory.Over Q
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (Ca …
          -/
          rw [Category.id_comp, ← pullback.condition, app_hom, Over.coe_hom]⟩⟩
          /-
            🎉 no goals
          -/


/-- A morphism that is surjective on pseudoelements is an epimorphism. -/
theorem epi_of_pseudo_surjective {P Q : C} (f : P ⟶ Q) : Function.Surjective f → Epi f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f) → C …
  -/
  intro h
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    ⊢ CategoryTheory.Epi f
  -/
  have ⟨pbar, hpbar⟩ := h (𝟙 Q)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    pbar : CategoryTheory.Abelian.Pseudoelement P
    hpbar : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f pbar) (Quot.mk  …
    ⊢ CategoryTheory.Epi f
  -/
  have ⟨p, hp⟩ := Quotient.exists_rep pbar
  have : (⟦(p.hom ≫ f : Over Q)⟧ : Quotient (setoid Q)) = ⟦↑(𝟙 Q)⟧ := by
    rw [← hp] at hpbar
    exact hpbar
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    pbar : CategoryTheory.Abelian.Pseudoelement P
    hpbar : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f pbar) (Quot.mk  …
    p : CategoryTheory.Over P
    hp : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid P) p) pbar
    this : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (Catego …
    ⊢ CategoryTheory.Epi f
  -/
  have ⟨R, x, y, _, ey, comm⟩ := Quotient.exact this
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    pbar : CategoryTheory.Abelian.Pseudoelement P
    hpbar : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f pbar) (Quot.mk  …
    p : CategoryTheory.Over P
    hp : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid P) p) pbar
    this : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (Catego …
    R : C
    x : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp p …
    y : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.id Q)) …
    w✝ : CategoryTheory.Epi x
    ey : CategoryTheory.Epi y
    comm : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.Over.mk (Categ …
    ⊢ CategoryTheory.Epi f
  -/
  apply @epi_of_epi_fac _ _ _ _ _ (x ≫ p.hom) f y ey
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    pbar : CategoryTheory.Abelian.Pseudoelement P
    hpbar : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f pbar) (Quot.mk  …
    p : CategoryTheory.Over P
    hp : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid P) p) pbar
    this : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (Catego …
    R : C
    x : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp p …
    y : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.id Q)) …
    w✝ : CategoryTheory.Epi x
    ey : CategoryTheory.Epi y
    comm : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.Over.mk (Categ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp x …
  -/
  dsimp at comm
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    pbar : CategoryTheory.Abelian.Pseudoelement P
    hpbar : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f pbar) (Quot.mk  …
    p : CategoryTheory.Over P
    hp : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid P) p) pbar
    this : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (Catego …
    R : C
    x : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp p …
    y : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.id Q)) …
    w✝ : CategoryTheory.Epi x
    ey : CategoryTheory.Epi y
    comm : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp x …
  -/
  rw [Category.assoc, comm]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Function.Surjective (CategoryTheory.Abelian.Pseudoelement.pseudoApply f)
    pbar : CategoryTheory.Abelian.Pseudoelement P
    hpbar : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f pbar) (Quot.mk  …
    p : CategoryTheory.Over P
    hp : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid P) p) pbar
    this : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (Catego …
    R : C
    x : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.comp p …
    y : Quiver.Hom R (CategoryTheory.Over.mk (CategoryTheory.CategoryStruct.id Q)) …
    w✝ : CategoryTheory.Epi x
    ey : CategoryTheory.Epi y
    comm : Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp y (CategoryTheory.CategoryStruct.id Q …
  -/
  apply Category.comp_id
  /-
    🎉 no goals
  -/


/-- Two morphisms in an exact sequence are exact on pseudoelements. -/
theorem pseudo_exact_of_exact {S : ShortComplex C} (hS : S.Exact) :
    ∀ b, S.g b = 0 → ∃ a, S.f a = b :=
  fun b' =>
    Quotient.inductionOn b' fun b hb => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : S.Exact
        b' : CategoryTheory.Abelian.Pseudoelement S.X₂
        b : CategoryTheory.Over S.X₂
        hb : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quotient.mk (Ca …
        ⊢ Exists fun a => Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f a)  …
      -/
      have hb' : b.hom ≫ S.g = 0 := (pseudoZero_iff _).1 hb
      -- By exactness, `b` factors through `im f = ker g` via some `c`.
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : S.Exact
        b' : CategoryTheory.Abelian.Pseudoelement S.X₂
        b : CategoryTheory.Over S.X₂
        hb : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quotient.mk (Ca …
        hb' : Eq (CategoryTheory.CategoryStruct.comp b.hom S.g) 0
        ⊢ Exists fun a => Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f a)  …
      -/
      obtain ⟨c, hc⟩ := KernelFork.IsLimit.lift' hS.isLimitImage _ hb'
      -- We compute the pullback of the map into the image and `c`.
      -- The pseudoelement induced by the first pullback map will be our preimage.
      /-
        case mk
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : S.Exact
        b' : CategoryTheory.Abelian.Pseudoelement S.X₂
        b : CategoryTheory.Over S.X₂
        hb : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quotient.mk (Ca …
        hb' : Eq (CategoryTheory.CategoryStruct.comp b.hom S.g) 0
        c : Quiver.Hom ((CategoryTheory.Functor.id C).obj b.left) (CategoryTheory.Limi …
        hc : Eq (CategoryTheory.CategoryStruct.comp c (CategoryTheory.Limits.Fork.ι (C …
        ⊢ Exists fun a => Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f a)  …
      -/
      use pullback.fst (Abelian.factorThruImage S.f) c
      -- It remains to show that the image of this element under `f` is pseudo-equal to `b`.
      /-
        case h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : S.Exact
        b' : CategoryTheory.Abelian.Pseudoelement S.X₂
        b : CategoryTheory.Over S.X₂
        hb : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quotient.mk (Ca …
        hb' : Eq (CategoryTheory.CategoryStruct.comp b.hom S.g) 0
        c : Quiver.Hom ((CategoryTheory.Functor.id C).obj b.left) (CategoryTheory.Limi …
        hc : Eq (CategoryTheory.CategoryStruct.comp c (CategoryTheory.Limits.Fork.ι (C …
        ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quot.mk (CategoryT …
      -/
      apply Quotient.sound
      refine ⟨pullback (Abelian.factorThruImage S.f) c, 𝟙 _,
              pullback.snd _ _, inferInstance, inferInstance, ?_⟩
      -- Now we can verify that the diagram commutes.
      calc
        𝟙 (pullback (Abelian.factorThruImage S.f) c) ≫ pullback.fst _ _ ≫ S.f =
          pullback.fst _ _ ≫ S.f :=
          Category.id_comp _
        _ = pullback.fst _ _ ≫ Abelian.factorThruImage S.f ≫ kernel.ι (cokernel.π S.f) := by
          rw [Abelian.image.fac]
        _ = (pullback.snd _ _ ≫ c) ≫ kernel.ι (cokernel.π S.f) := by
          rw [← Category.assoc, pullback.condition]
        _ = pullback.snd _ _ ≫ b.hom := by
          rw [Category.assoc]
          congr


theorem apply_eq_zero_of_comp_eq_zero {P Q R : C} (f : Q ⟶ R) (a : P ⟶ Q) : a ≫ f = 0 → f a = 0 :=
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                inst✝ : CategoryTheory.Abelian C
                P Q R : C
                f : Quiver.Hom Q R
                a : Quiver.Hom P Q
                h : Eq (CategoryTheory.CategoryStruct.comp a f) 0
                ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quot.mk (CategoryThe …
              -/
  fun h => by simp [over_coe_def, pseudoApply_mk', Over.coe_hom, h]
              /-
                🎉 no goals
              -/


/-- If two morphisms are exact on pseudoelements, they are exact. -/
theorem exact_of_pseudo_exact (S : ShortComplex C)
    (hS : ∀ b, S.g b = 0 → ∃ a, S.f a = b) : S.Exact :=
  (S.exact_iff_kernel_ι_comp_cokernel_π_zero).2 (by
      -- If we apply `g` to the pseudoelement induced by its kernel, we get 0 (of course!).
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      have : S.g (kernel.ι S.g) = 0 := apply_eq_zero_of_comp_eq_zero _ _ (kernel.condition _)
      -- By pseudo-exactness, we get a preimage.
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      obtain ⟨a', ha⟩ := hS _ this
      /-
        case intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f a') (Quot.mk (Ca …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      obtain ⟨a, ha'⟩ := Quotient.exists_rep a'
      /-
        case intro.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f a') (Quot.mk (Ca …
        a : CategoryTheory.Over S.X₁
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      rw [← ha'] at ha
      /-
        case intro.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        a : CategoryTheory.Over S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quotient.mk (Ca …
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      obtain ⟨Z, r, q, _, eq, comm⟩ := Quotient.exact ha
      -- Consider the pullback of `kernel.ι (cokernel.π f)` and `kernel.ι g`.
      -- The commutative diagram given by the pseudo-equality `f a = b` induces
      -- a cone over this pullback, so we get a factorization `z`.
      obtain ⟨z, _, hz₂⟩ := @pullback.lift' _ _ _ _ _ _ (kernel.ι (cokernel.π S.f))
        (kernel.ι S.g) _ (r ≫ a.hom ≫ Abelian.factorThruImage S.f) q (by
          simp only [Category.assoc, Abelian.image.fac]
          exact comm)
      -- Let's give a name to the second pullback morphism.
      /-
        case intro.intro.intro.intro.intro.intro.intro.mk.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        a : CategoryTheory.Over S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quotient.mk (Ca …
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        Z : C
        r : Quiver.Hom Z ((fun g => CategoryTheory.Abelian.app S.f g) a).left
        q : Quiver.Hom Z (CategoryTheory.Over.mk (CategoryTheory.Limits.kernel.ι S.g)) …
        w✝ : CategoryTheory.Epi r
        eq : CategoryTheory.Epi q
        comm : Eq (CategoryTheory.CategoryStruct.comp r ((fun g => CategoryTheory.Abel …
        z : Quiver.Hom Z (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullba …
        hz₂ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullback …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      let j : pullback (kernel.ι (cokernel.π S.f)) (kernel.ι S.g) ⟶ kernel S.g := pullback.snd _ _
      -- Since `q` is an epimorphism, in particular this means that `j` is an epimorphism.
      /-
        case intro.intro.intro.intro.intro.intro.intro.mk.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        a : CategoryTheory.Over S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quotient.mk (Ca …
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        Z : C
        r : Quiver.Hom Z ((fun g => CategoryTheory.Abelian.app S.f g) a).left
        q : Quiver.Hom Z (CategoryTheory.Over.mk (CategoryTheory.Limits.kernel.ι S.g)) …
        w✝ : CategoryTheory.Epi r
        eq : CategoryTheory.Epi q
        comm : Eq (CategoryTheory.CategoryStruct.comp r ((fun g => CategoryTheory.Abel …
        z : Quiver.Hom Z (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullba …
        hz₂ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullback …
        j : Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel.ι …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      haveI pe : Epi j := epi_of_epi_fac hz₂
      -- But it is also a monomorphism, because `kernel.ι (cokernel.π f)` is: A kernel is
      -- always a monomorphism and the pullback of a monomorphism is a monomorphism.
      -- But mono + epi = iso, so `j` is an isomorphism.
      /-
        case intro.intro.intro.intro.intro.intro.intro.mk.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cate …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        a : CategoryTheory.Over S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quotient.mk (Ca …
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        Z : C
        r : Quiver.Hom Z ((fun g => CategoryTheory.Abelian.app S.f g) a).left
        q : Quiver.Hom Z (CategoryTheory.Over.mk (CategoryTheory.Limits.kernel.ι S.g)) …
        w✝ : CategoryTheory.Epi r
        eq : CategoryTheory.Epi q
        comm : Eq (CategoryTheory.CategoryStruct.comp r ((fun g => CategoryTheory.Abel …
        z : Quiver.Hom Z (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullba …
        hz₂ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullback …
        j : Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel.ι …
        pe : CategoryTheory.Epi j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      haveI : IsIso j := isIso_of_mono_of_epi _
      -- But then `kernel.ι g` can be expressed using all of the maps of the pullback square, and we
      -- are done.
      /-
        case intro.intro.intro.intro.intro.intro.intro.mk.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this✝ : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cat …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        a : CategoryTheory.Over S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quotient.mk (Ca …
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        Z : C
        r : Quiver.Hom Z ((fun g => CategoryTheory.Abelian.app S.f g) a).left
        q : Quiver.Hom Z (CategoryTheory.Over.mk (CategoryTheory.Limits.kernel.ι S.g)) …
        w✝ : CategoryTheory.Epi r
        eq : CategoryTheory.Epi q
        comm : Eq (CategoryTheory.CategoryStruct.comp r ((fun g => CategoryTheory.Abel …
        z : Quiver.Hom Z (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullba …
        hz₂ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullback …
        j : Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel.ι …
        pe : CategoryTheory.Epi j
        this : CategoryTheory.IsIso j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
      -/
      rw [(Iso.eq_inv_comp (asIso j)).2 pullback.condition.symm]
      /-
        case intro.intro.intro.intro.intro.intro.intro.mk.intro
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Abelian C
        S : CategoryTheory.ShortComplex C
        hS : ∀ (b : CategoryTheory.Abelian.Pseudoelement S.X₂), Eq (CategoryTheory.Abe …
        this✝ : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.g (Quot.mk (Cat …
        a' : CategoryTheory.Abelian.Pseudoelement S.X₁
        a : CategoryTheory.Over S.X₁
        ha : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply S.f (Quotient.mk (Ca …
        ha' : Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid S.X₁) a) a'
        Z : C
        r : Quiver.Hom Z ((fun g => CategoryTheory.Abelian.app S.f g) a).left
        q : Quiver.Hom Z (CategoryTheory.Over.mk (CategoryTheory.Limits.kernel.ι S.g)) …
        w✝ : CategoryTheory.Epi r
        eq : CategoryTheory.Epi q
        comm : Eq (CategoryTheory.CategoryStruct.comp r ((fun g => CategoryTheory.Abel …
        z : Quiver.Hom Z (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel …
        left✝ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullba …
        hz₂ : Eq (CategoryTheory.CategoryStruct.comp z (CategoryTheory.Limits.pullback …
        j : Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.Limits.kernel.ι …
        pe : CategoryTheory.Epi j
        this : CategoryTheory.IsIso j
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
      simp only [Category.assoc, kernel.condition, HasZeroMorphisms.comp_zero])
      /-
        🎉 no goals
      -/


/-- If two pseudoelements `x` and `y` have the same image under some morphism `f`, then we can form
    their "difference" `z`. This pseudoelement has the properties that `f z = 0` and for all
    morphisms `g`, if `g y = 0` then `g z = g x`. -/
theorem sub_of_eq_image {P Q : C} (f : P ⟶ Q) (x y : P) :
    f x = f y → ∃ z, f z = 0 ∧ ∀ (R : C) (g : P ⟶ R), (g : P ⟶ R) y = 0 → g z = g x :=
  Quotient.inductionOn₂ x y fun a a' h =>
    match Quotient.exact h with
    | ⟨R, p, q, ep, _, comm⟩ =>
      let a'' : R ⟶ P := ↑(p ≫ a.hom) - ↑(q ≫ a'.hom)
      ⟨a'',
        ⟨show ⟦(a'' ≫ f : Over Q)⟧ = ⟦↑(0 : Q ⟶ Q)⟧ by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Abelian C
              P Q : C
              f : Quiver.Hom P Q
              x y : CategoryTheory.Abelian.Pseudoelement P
              a a' : CategoryTheory.Over P
              h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
              R : C
              p : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a).left
              q : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a').left
              ep : CategoryTheory.Epi p
              w✝ : CategoryTheory.Epi q
              comm : Eq (CategoryTheory.CategoryStruct.comp p ((fun g => CategoryTheory.Abel …
              a'' : Quiver.Hom R P := HSub.hSub (CategoryTheory.CategoryStruct.comp p a.hom) …
              ⊢ Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (CategoryThe …
            -/
            dsimp at comm
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              inst✝ : CategoryTheory.Abelian C
              P Q : C
              f : Quiver.Hom P Q
              x y : CategoryTheory.Abelian.Pseudoelement P
              a a' : CategoryTheory.Over P
              h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
              R : C
              p : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a).left
              q : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a').left
              ep : CategoryTheory.Epi p
              w✝ : CategoryTheory.Epi q
              comm : Eq (CategoryTheory.CategoryStruct.comp p (CategoryTheory.CategoryStruct …
              a'' : Quiver.Hom R P := HSub.hSub (CategoryTheory.CategoryStruct.comp p a.hom) …
              ⊢ Eq (Quotient.mk (CategoryTheory.Abelian.Pseudoelement.setoid Q) (CategoryThe …
            -/
            simp [a'', sub_eq_zero.2 comm],
            /-
              🎉 no goals
            -/
          fun Z g hh => by
          /-
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Abelian C
            P Q : C
            f : Quiver.Hom P Q
            x y : CategoryTheory.Abelian.Pseudoelement P
            a a' : CategoryTheory.Over P
            h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
            R : C
            p : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a).left
            q : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a').left
            ep : CategoryTheory.Epi p
            w✝ : CategoryTheory.Epi q
            comm : Eq (CategoryTheory.CategoryStruct.comp p ((fun g => CategoryTheory.Abel …
            a'' : Quiver.Hom R P := HSub.hSub (CategoryTheory.CategoryStruct.comp p a.hom) …
            Z : C
            g : Quiver.Hom P Z
            hh : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply g (Quotient.mk (Cate …
            ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply g (Quot.mk (CategoryThe …
          -/
          obtain ⟨X, p', q', ep', _, comm'⟩ := Quotient.exact hh
          have : a'.hom ≫ g = 0 := by
            apply (epi_iff_cancel_zero _).1 ep' _ (a'.hom ≫ g)
            simpa using comm'
          /-
            case intro.intro.intro.intro.intro
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Abelian C
            P Q : C
            f : Quiver.Hom P Q
            x y : CategoryTheory.Abelian.Pseudoelement P
            a a' : CategoryTheory.Over P
            h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
            R : C
            p : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a).left
            q : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a').left
            ep : CategoryTheory.Epi p
            w✝¹ : CategoryTheory.Epi q
            comm : Eq (CategoryTheory.CategoryStruct.comp p ((fun g => CategoryTheory.Abel …
            a'' : Quiver.Hom R P := HSub.hSub (CategoryTheory.CategoryStruct.comp p a.hom) …
            Z : C
            g : Quiver.Hom P Z
            hh : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply g (Quotient.mk (Cate …
            X : C
            p' : Quiver.Hom X ((fun g_1 => CategoryTheory.Abelian.app g g_1) a').left
            q' : Quiver.Hom X (CategoryTheory.Over.mk 0).left
            ep' : CategoryTheory.Epi p'
            w✝ : CategoryTheory.Epi q'
            comm' : Eq (CategoryTheory.CategoryStruct.comp p' ((fun g_1 => CategoryTheory. …
            this : Eq (CategoryTheory.CategoryStruct.comp a'.hom g) 0
            ⊢ Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply g (Quot.mk (CategoryThe …
          -/
          apply Quotient.sound
          -- Can we prevent quotient.sound from giving us this weird `coe_b` thingy?
          /-
            case intro.intro.intro.intro.intro.a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Abelian C
            P Q : C
            f : Quiver.Hom P Q
            x y : CategoryTheory.Abelian.Pseudoelement P
            a a' : CategoryTheory.Over P
            h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
            R : C
            p : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a).left
            q : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a').left
            ep : CategoryTheory.Epi p
            w✝¹ : CategoryTheory.Epi q
            comm : Eq (CategoryTheory.CategoryStruct.comp p ((fun g => CategoryTheory.Abel …
            a'' : Quiver.Hom R P := HSub.hSub (CategoryTheory.CategoryStruct.comp p a.hom) …
            Z : C
            g : Quiver.Hom P Z
            hh : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply g (Quotient.mk (Cate …
            X : C
            p' : Quiver.Hom X ((fun g_1 => CategoryTheory.Abelian.app g g_1) a').left
            q' : Quiver.Hom X (CategoryTheory.Over.mk 0).left
            ep' : CategoryTheory.Epi p'
            w✝ : CategoryTheory.Epi q'
            comm' : Eq (CategoryTheory.CategoryStruct.comp p' ((fun g_1 => CategoryTheory. …
            this : Eq (CategoryTheory.CategoryStruct.comp a'.hom g) 0
            ⊢ HasEquiv.Equiv ((fun g_1 => CategoryTheory.Abelian.app g g_1) (CategoryTheor …
          -/
          change app g (a'' : Over P) ≈ app g a
          /-
            case intro.intro.intro.intro.intro.a
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            inst✝ : CategoryTheory.Abelian C
            P Q : C
            f : Quiver.Hom P Q
            x y : CategoryTheory.Abelian.Pseudoelement P
            a a' : CategoryTheory.Over P
            h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
            R : C
            p : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a).left
            q : Quiver.Hom R ((fun g => CategoryTheory.Abelian.app f g) a').left
            ep : CategoryTheory.Epi p
            w✝¹ : CategoryTheory.Epi q
            comm : Eq (CategoryTheory.CategoryStruct.comp p ((fun g => CategoryTheory.Abel …
            a'' : Quiver.Hom R P := HSub.hSub (CategoryTheory.CategoryStruct.comp p a.hom) …
            Z : C
            g : Quiver.Hom P Z
            hh : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply g (Quotient.mk (Cate …
            X : C
            p' : Quiver.Hom X ((fun g_1 => CategoryTheory.Abelian.app g g_1) a').left
            q' : Quiver.Hom X (CategoryTheory.Over.mk 0).left
            ep' : CategoryTheory.Epi p'
            w✝ : CategoryTheory.Epi q'
            comm' : Eq (CategoryTheory.CategoryStruct.comp p' ((fun g_1 => CategoryTheory. …
            this : Eq (CategoryTheory.CategoryStruct.comp a'.hom g) 0
            ⊢ HasEquiv.Equiv (CategoryTheory.Abelian.app g (CategoryTheory.Over.mk a'')) ( …
          -/
          exact ⟨R, 𝟙 R, p, inferInstance, ep, by simp [a'', sub_eq_add_neg, this]⟩⟩⟩
          /-
            🎉 no goals
          -/


/-- If `f : P ⟶ R` and `g : Q ⟶ R` are morphisms and `p : P` and `q : Q` are pseudoelements such
    that `f p = g q`, then there is some `s : pullback f g` such that `fst s = p` and `snd s = q`.

    Remark: Borceux claims that `s` is unique, but this is false. See
    `Counterexamples/Pseudoelement.lean` for details. -/
theorem pseudo_pullback {P Q R : C} {f : P ⟶ R} {g : Q ⟶ R} {p : P} {q : Q} :
    f p = g q →
      ∃ s, pullback.fst f g s = p ∧ pullback.snd f g s = q :=
  Quotient.inductionOn₂ p q fun x y h => by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      P Q R : C
      f : Quiver.Hom P R
      g : Quiver.Hom Q R
      p : CategoryTheory.Abelian.Pseudoelement P
      q : CategoryTheory.Abelian.Pseudoelement Q
      x : CategoryTheory.Over P
      y : CategoryTheory.Over Q
      h : Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply f (Quotient.mk (Categ …
      ⊢ Exists fun s => And (Eq (CategoryTheory.Abelian.Pseudoelement.pseudoApply (C …
    -/
    obtain ⟨Z, a, b, ea, eb, comm⟩ := Quotient.exact h
    obtain ⟨l, hl₁, hl₂⟩ := @pullback.lift' _ _ _ _ _ _ f g _ (a ≫ x.hom) (b ≫ y.hom) (by
      simp only [Category.assoc]
      exact comm)
    exact ⟨l, ⟨Quotient.sound ⟨Z, 𝟙 Z, a, inferInstance, ea, by rwa [Category.id_comp]⟩,
      Quotient.sound ⟨Z, 𝟙 Z, b, inferInstance, eb, by rwa [Category.id_comp]⟩⟩⟩


/-- In the category `Module R`, if `x` and `y` are pseudoequal, then the range of the associated
morphisms is the same. -/
theorem ModuleCat.eq_range_of_pseudoequal {R : Type*} [CommRing R] {G : ModuleCat R} {x y : Over G}
    (h : PseudoEqual G x y) : LinearMap.range x.hom.hom = LinearMap.range y.hom.hom := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    G : ModuleCat R
    x y : CategoryTheory.Over G
    h : CategoryTheory.Abelian.PseudoEqual G x y
    ⊢ Eq (LinearMap.range x.hom.hom) (LinearMap.range y.hom.hom)
  -/
  obtain ⟨P, p, q, hp, hq, H⟩ := h
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝ : CommRing R
    G : ModuleCat R
    x y : CategoryTheory.Over G
    P : ModuleCat R
    p : Quiver.Hom P x.left
    q : Quiver.Hom P y.left
    hp : CategoryTheory.Epi p
    hq : CategoryTheory.Epi q
    H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
    ⊢ Eq (LinearMap.range x.hom.hom) (LinearMap.range y.hom.hom)
  -/
  refine Submodule.ext fun a => ⟨fun ha => ?_, fun ha => ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      ha : Membership.mem (LinearMap.range x.hom.hom) a
      ⊢ Membership.mem (LinearMap.range y.hom.hom) a
    -/
  · obtain ⟨a', ha'⟩ := ha
    /-
      case intro.intro.intro.intro.intro.refine_1.intro
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      a' : ↑((CategoryTheory.Functor.id (ModuleCat R)).obj x.left)
      ha' : Eq (x.hom.hom a') a
      ⊢ Membership.mem (LinearMap.range y.hom.hom) a
    -/
    obtain ⟨a'', ha''⟩ := (ModuleCat.epi_iff_surjective p).1 hp a'
    /-
      case intro.intro.intro.intro.intro.refine_1.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      a' : ↑((CategoryTheory.Functor.id (ModuleCat R)).obj x.left)
      ha' : Eq (x.hom.hom a') a
      a'' : ↑P
      ha'' : Eq (p.hom a'') a'
      ⊢ Membership.mem (LinearMap.range y.hom.hom) a
    -/
    refine ⟨q a'', ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      a' : ↑((CategoryTheory.Functor.id (ModuleCat R)).obj x.left)
      ha' : Eq (x.hom.hom a') a
      a'' : ↑P
      ha'' : Eq (p.hom a'') a'
      ⊢ Eq (y.hom.hom (q.hom a'')) a
    -/
    dsimp at ha' ⊢
    rw [← LinearMap.comp_apply, ← ModuleCat.hom_comp, ← H,
      ModuleCat.hom_comp, LinearMap.comp_apply, ha'', ha']
    /-
      case intro.intro.intro.intro.intro.refine_2
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      ha : Membership.mem (LinearMap.range y.hom.hom) a
      ⊢ Membership.mem (LinearMap.range x.hom.hom) a
    -/
  · obtain ⟨a', ha'⟩ := ha
    /-
      case intro.intro.intro.intro.intro.refine_2.intro
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      a' : ↑((CategoryTheory.Functor.id (ModuleCat R)).obj y.left)
      ha' : Eq (y.hom.hom a') a
      ⊢ Membership.mem (LinearMap.range x.hom.hom) a
    -/
    obtain ⟨a'', ha''⟩ := (ModuleCat.epi_iff_surjective q).1 hq a'
    /-
      case intro.intro.intro.intro.intro.refine_2.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      a' : ↑((CategoryTheory.Functor.id (ModuleCat R)).obj y.left)
      ha' : Eq (y.hom.hom a') a
      a'' : ↑P
      ha'' : Eq (q.hom a'') a'
      ⊢ Membership.mem (LinearMap.range x.hom.hom) a
    -/
    refine ⟨p a'', ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_2.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      G : ModuleCat R
      x y : CategoryTheory.Over G
      P : ModuleCat R
      p : Quiver.Hom P x.left
      q : Quiver.Hom P y.left
      hp : CategoryTheory.Epi p
      hq : CategoryTheory.Epi q
      H : Eq (CategoryTheory.CategoryStruct.comp p x.hom) (CategoryTheory.CategorySt …
      a : ↑((CategoryTheory.Functor.fromPUnit G).obj x.right)
      a' : ↑((CategoryTheory.Functor.id (ModuleCat R)).obj y.left)
      ha' : Eq (y.hom.hom a') a
      a'' : ↑P
      ha'' : Eq (q.hom a'') a'
      ⊢ Eq (x.hom.hom (p.hom a'')) a
    -/
    dsimp at ha' ⊢
    rw [← LinearMap.comp_apply, ← ModuleCat.hom_comp, H, ModuleCat.hom_comp, LinearMap.comp_apply,
      ha'', ha']


