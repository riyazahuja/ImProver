local notation "β" => ofTypeMonad Ultrafilter


/-- The type `Compactum` of Compacta, defined as algebras for the ultrafilter monad. -/
def Compactum :=
  Monad.Algebra β deriving Category, Inhabited


/-- The forgetful functor to Type* -/
def forget : Compactum ⥤ Type* :=
  Monad.forget _ --deriving CreatesLimits, Faithful
  -- Porting note: deriving fails, adding manually. Note `CreatesLimits` now noncomputable


instance : forget.Faithful :=
  show (Monad.forget _).Faithful from inferInstance


noncomputable instance : CreatesLimits forget :=
  show CreatesLimits <| Monad.forget _ from inferInstance


/-- The "free" Compactum functor. -/
def free : Type* ⥤ Compactum :=
  Monad.free _


/-- The adjunction between `free` and `forget`. -/
def adj : free ⊣ forget :=
  Monad.adj _

-- Basic instances

instance : ConcreteCategory Compactum where forget := forget

-- Porting note: changed from forget to X.A

instance : CoeSort Compactum Type* :=
  ⟨fun X => X.A⟩


instance {X Y : Compactum} : CoeFun (X ⟶ Y) fun _ => X → Y :=
  ⟨fun f => f.f⟩


instance : HasLimits Compactum :=
  hasLimits_of_hasLimits_createsLimits forget


/-- The structure map for a compactum, essentially sending an ultrafilter to its limit. -/
def str (X : Compactum) : Ultrafilter X → X :=
  X.a


/-- The monadic join. -/
def join (X : Compactum) : Ultrafilter (Ultrafilter X) → Ultrafilter X :=
  (β ).μ.app _


/-- The inclusion of `X` into `Ultrafilter X`. -/
def incl (X : Compactum) : X → Ultrafilter X :=
  (β ).η.app _


@[simp]
theorem str_incl (X : Compactum) (x : X) : X.str (X.incl x) = x := by
  /-
    X : Compactum
    x : X.A
    ⊢ Eq (X.str (X.incl x)) x
  -/
  change ((β ).η.app _ ≫ X.a) _ = _
  /-
    X : Compactum
    x : X.A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
  -/
  rw [Monad.Algebra.unit]
  /-
    X : Compactum
    x : X.A
    ⊢ Eq (CategoryTheory.CategoryStruct.id X.A x) x
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem str_hom_commute (X Y : Compactum) (f : X ⟶ Y) (xs : Ultrafilter X) :
    f (X.str xs) = Y.str (map f xs) := by
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    xs : Ultrafilter X.A
    ⊢ Eq (f.f (X.str xs)) (Y.str (Ultrafilter.map f.f xs))
  -/
  change (X.a ≫ f.f) _ = _
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    xs : Ultrafilter X.A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp X.a f.f xs) (Y.str (Ultrafilter.map f …
  -/
  rw [← f.h]
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    xs : Ultrafilter X.A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem join_distrib (X : Compactum) (uux : Ultrafilter (Ultrafilter X)) :
    X.str (X.join uux) = X.str (map X.str uux) := by
  /-
    X : Compactum
    uux : Ultrafilter (Ultrafilter X.A)
    ⊢ Eq (X.str (X.join uux)) (X.str (Ultrafilter.map X.str uux))
  -/
  change ((β ).μ.app _ ≫ X.a) _ = _
  /-
    X : Compactum
    uux : Ultrafilter (Ultrafilter X.A)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
  -/
  rw [Monad.Algebra.assoc]
  /-
    X : Compactum
    uux : Ultrafilter (Ultrafilter X.A)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: changes to X.A from X since Lean can't see through X to X.A below

instance {X : Compactum} : TopologicalSpace X.A where
  IsOpen U := ∀ F : Ultrafilter X, X.str F ∈ U → U ∈ F
  isOpen_univ _ _ := Filter.univ_sets _
  isOpen_inter _ _ h3 h4 _ h6 := Filter.inter_sets _ (h3 _ h6.1) (h4 _ h6.2)
  isOpen_sUnion := fun _ h1 _ ⟨T, hT, h2⟩ =>
    mem_of_superset (h1 T hT _ h2) (Set.subset_sUnion_of_mem hT)


theorem isClosed_iff {X : Compactum} (S : Set X) :
    IsClosed S ↔ ∀ F : Ultrafilter X, S ∈ F → X.str F ∈ S := by
  /-
    X : Compactum
    S : Set X.A
    ⊢ Iff (IsClosed S) (∀ (F : Ultrafilter X.A), Membership.mem F S → Membership.m …
  -/
  rw [← isOpen_compl_iff]
  /-
    X : Compactum
    S : Set X.A
    ⊢ Iff (IsOpen (HasCompl.compl S)) (∀ (F : Ultrafilter X.A), Membership.mem F S …
  -/
  constructor
    /-
      case mp
      X : Compactum
      S : Set X.A
      ⊢ IsOpen (HasCompl.compl S) → ∀ (F : Ultrafilter X.A), Membership.mem F S → Me …
    -/
  · intro cond F h
    /-
      case mp
      X : Compactum
      S : Set X.A
      cond : IsOpen (HasCompl.compl S)
      F : Ultrafilter X.A
      h : Membership.mem F S
      ⊢ Membership.mem S (X.str F)
    -/
    by_contra c
    /-
      case mp
      X : Compactum
      S : Set X.A
      cond : IsOpen (HasCompl.compl S)
      F : Ultrafilter X.A
      h : Membership.mem F S
      c : Not (Membership.mem S (X.str F))
      ⊢ False
    -/
    specialize cond F c
    /-
      case mp
      X : Compactum
      S : Set X.A
      F : Ultrafilter X.A
      h : Membership.mem F S
      c : Not (Membership.mem S (X.str F))
      cond : Membership.mem F (HasCompl.compl S)
      ⊢ False
    -/
    rw [compl_mem_iff_not_mem] at cond
    /-
      case mp
      X : Compactum
      S : Set X.A
      F : Ultrafilter X.A
      h : Membership.mem F S
      c : Not (Membership.mem S (X.str F))
      cond : Not (Membership.mem F S)
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Compactum
      S : Set X.A
      ⊢ (∀ (F : Ultrafilter X.A), Membership.mem F S → Membership.mem S (X.str F)) → …
    -/
  · intro h1 F h2
    /-
      case mpr
      X : Compactum
      S : Set X.A
      h1 : ∀ (F : Ultrafilter X.A), Membership.mem F S → Membership.mem S (X.str F)
      F : Ultrafilter X.A
      h2 : Membership.mem (HasCompl.compl S) (X.str F)
      ⊢ Membership.mem F (HasCompl.compl S)
    -/
    specialize h1 F
    /-
      case mpr
      X : Compactum
      S : Set X.A
      F : Ultrafilter X.A
      h2 : Membership.mem (HasCompl.compl S) (X.str F)
      h1 : Membership.mem F S → Membership.mem S (X.str F)
      ⊢ Membership.mem F (HasCompl.compl S)
    -/
    cases' F.mem_or_compl_mem S with h h
    /-
      case mpr.inl
      X : Compactum
      S : Set X.A
      F : Ultrafilter X.A
      h2 : Membership.mem (HasCompl.compl S) (X.str F)
      h1 : Membership.mem F S → Membership.mem S (X.str F)
      h : Membership.mem F S
      ⊢ Membership.mem F (HasCompl.compl S)
    -/
    exacts [absurd (h1 h) h2, h]
    /-
      🎉 no goals
    -/


instance {X : Compactum} : CompactSpace X := by
  /-
    X : Compactum
    ⊢ CompactSpace X.A
  -/
  constructor
  /-
    case isCompact_univ
    X : Compactum
    ⊢ IsCompact Set.univ
  -/
  rw [isCompact_iff_ultrafilter_le_nhds]
  /-
    case isCompact_univ
    X : Compactum
    ⊢ ∀ (f : Ultrafilter X.A), LE.le (↑f) (Filter.principal Set.univ) → Exists fun …
  -/
  intro F _
  /-
    case isCompact_univ
    X : Compactum
    F : Ultrafilter X.A
    a✝ : LE.le (↑F) (Filter.principal Set.univ)
    ⊢ Exists fun x => And (Membership.mem Set.univ x) (LE.le (↑F) (nhds x))
  -/
  refine ⟨X.str F, by tauto, ?_⟩
  /-
    case isCompact_univ
    X : Compactum
    F : Ultrafilter X.A
    a✝ : LE.le (↑F) (Filter.principal Set.univ)
    ⊢ LE.le (↑F) (nhds (X.str F))
  -/
  rw [le_nhds_iff]
  /-
    case isCompact_univ
    X : Compactum
    F : Ultrafilter X.A
    a✝ : LE.le (↑F) (Filter.principal Set.univ)
    ⊢ ∀ (s : Set X.A), Membership.mem s (X.str F) → IsOpen s → Membership.mem (↑F) s
  -/
  intro S h1 h2
  /-
    case isCompact_univ
    X : Compactum
    F : Ultrafilter X.A
    a✝ : LE.le (↑F) (Filter.principal Set.univ)
    S : Set X.A
    h1 : Membership.mem S (X.str F)
    h2 : IsOpen S
    ⊢ Membership.mem (↑F) S
  -/
  exact h2 F h1
  /-
    🎉 no goals
  -/


/-- A local definition used only in the proofs. -/
private def basic {X : Compactum} (A : Set X) : Set (Ultrafilter X) :=
  { F | A ∈ F }


/-- A local definition used only in the proofs. -/
private def cl {X : Compactum} (A : Set X) : Set X :=
  X.str '' basic A


private theorem basic_inter {X : Compactum} (A B : Set X) : basic (A ∩ B) = basic A ∩ basic B := by
  /-
    X : Compactum
    A B : Set X.A
    ⊢ Eq (Compactum.basic (Inter.inter A B)) (Inter.inter (Compactum.basic A) (Com …
  -/
  ext G
  /-
    case h
    X : Compactum
    A B : Set X.A
    G : Ultrafilter X.A
    ⊢ Iff (Membership.mem (Compactum.basic (Inter.inter A B)) G) (Membership.mem ( …
  -/
  constructor
    /-
      case h.mp
      X : Compactum
      A B : Set X.A
      G : Ultrafilter X.A
      ⊢ Membership.mem (Compactum.basic (Inter.inter A B)) G → Membership.mem (Inter …
    -/
  · intro hG
    /-
      case h.mp
      X : Compactum
      A B : Set X.A
      G : Ultrafilter X.A
      hG : Membership.mem (Compactum.basic (Inter.inter A B)) G
      ⊢ Membership.mem (Inter.inter (Compactum.basic A) (Compactum.basic B)) G
    -/
    constructor <;> filter_upwards [hG] with _
    /-
      case h
      X : Compactum
      A B : Set X.A
      G : Ultrafilter X.A
      hG : Membership.mem (Compactum.basic (Inter.inter A B)) G
      a✝ : X.A
      ⊢ Membership.mem (Inter.inter A B) a✝ → Membership.mem A a✝
    -/
    exacts [And.left, And.right]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : Compactum
      A B : Set X.A
      G : Ultrafilter X.A
      ⊢ Membership.mem (Inter.inter (Compactum.basic A) (Compactum.basic B)) G → Mem …
    -/
  · rintro ⟨h1, h2⟩
    /-
      case h.mpr.intro
      X : Compactum
      A B : Set X.A
      G : Ultrafilter X.A
      h1 : Membership.mem (Compactum.basic A) G
      h2 : Membership.mem (Compactum.basic B) G
      ⊢ Membership.mem (Compactum.basic (Inter.inter A B)) G
    -/
    exact inter_mem h1 h2
    /-
      🎉 no goals
    -/


private theorem subset_cl {X : Compactum} (A : Set X) : A ⊆ cl A := fun a ha =>
                    /-
                      X : Compactum
                      A : Set X.A
                      a : X.A
                      ha : Membership.mem A a
                      ⊢ Eq (X.str (X.incl a)) a
                    -/
  ⟨X.incl a, ha, by simp⟩
                    /-
                      🎉 no goals
                    -/


private theorem cl_cl {X : Compactum} (A : Set X) : cl (cl A) ⊆ cl A := by
  /-
    X : Compactum
    A : Set X.A
    ⊢ HasSubset.Subset (Compactum.cl (Compactum.cl A)) (Compactum.cl A)
  -/
  rintro _ ⟨F, hF, rfl⟩
  -- Notation to be used in this proof.
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let fsu := Finset (Set (Ultrafilter X))
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let ssu := Set (Set (Ultrafilter X))
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let ι : fsu → ssu := fun x ↦ ↑x
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let C0 : ssu := { Z | ∃ B ∈ F, X.str ⁻¹' B = Z }
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let AA := { G : Ultrafilter X | A ∈ G }
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let C1 := insert AA C0
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  let C2 := finiteInterClosure C1
  -- C0 is closed under intersections.
  have claim1 : ∀ (B) (_ : B ∈ C0) (C) (_ : C ∈ C0), B ∩ C ∈ C0 := by
    rintro B ⟨Q, hQ, rfl⟩ C ⟨R, hR, rfl⟩
    use Q ∩ R
    simp only [and_true, eq_self_iff_true, Set.preimage_inter]
    exact inter_sets _ hQ hR
  -- All sets in C0 are nonempty.
  have claim2 : ∀ B ∈ C0, Set.Nonempty B := by
    rintro B ⟨Q, hQ, rfl⟩
    obtain ⟨q⟩ := Filter.nonempty_of_mem hQ
    use X.incl q
    simpa
  -- The intersection of AA with every set in C0 is nonempty.
  have claim3 : ∀ B ∈ C0, (AA ∩ B).Nonempty := by
    rintro B ⟨Q, hQ, rfl⟩
    have : (Q ∩ cl A).Nonempty := Filter.nonempty_of_mem (inter_mem hQ hF)
    rcases this with ⟨q, hq1, P, hq2, hq3⟩
    refine ⟨P, hq2, ?_⟩
    rw [← hq3] at hq1
    simpa
  -- Suffices to show that the intersection of any finite subcollection of C1 is nonempty.
  suffices ∀ T : fsu, ι T ⊆ C1 → (⋂₀ ι T).Nonempty by
    obtain ⟨G, h1⟩ := exists_ultrafilter_of_finite_inter_nonempty _ this
    use X.join G
    have : G.map X.str = F := Ultrafilter.coe_le_coe.1 fun S hS => h1 (Or.inr ⟨S, hS, rfl⟩)
    rw [join_distrib, this]
    exact ⟨h1 (Or.inl rfl), rfl⟩
  -- C2 is closed under finite intersections (by construction!).
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    ⊢ ∀ (T : fsu), HasSubset.Subset (ι T) C1 → (Set.sInter (ι T)).Nonempty
  -/
  have claim4 := finiteInterClosure_finiteInter C1
  -- C0 is closed under finite intersections by claim1.
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    claim4 : FiniteInter (FiniteInter.finiteInterClosure C1)
    ⊢ ∀ (T : fsu), HasSubset.Subset (ι T) C1 → (Set.sInter (ι T)).Nonempty
  -/
  have claim5 : FiniteInter C0 := ⟨⟨_, univ_mem, Set.preimage_univ⟩, claim1⟩
  -- Every element of C2 is nonempty.
  have claim6 : ∀ P ∈ C2, (P : Set (Ultrafilter X)).Nonempty := by
    suffices ∀ P ∈ C2, P ∈ C0 ∨ ∃ Q ∈ C0, P = AA ∩ Q by
      intro P hP
      cases' this P hP with h h
      · exact claim2 _ h
      · rcases h with ⟨Q, hQ, rfl⟩
        exact claim3 _ hQ
    intro P hP
    exact claim5.finiteInterClosure_insert _ hP
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    claim4 : FiniteInter (FiniteInter.finiteInterClosure C1)
    claim5 : FiniteInter C0
    claim6 : ∀ (P : Set (Ultrafilter X.A)), Membership.mem C2 P → P.Nonempty
    ⊢ ∀ (T : fsu), HasSubset.Subset (ι T) C1 → (Set.sInter (ι T)).Nonempty
  -/
  intro T hT
  -- Suffices to show that the intersection of the T's is contained in C2.
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    claim4 : FiniteInter (FiniteInter.finiteInterClosure C1)
    claim5 : FiniteInter C0
    claim6 : ∀ (P : Set (Ultrafilter X.A)), Membership.mem C2 P → P.Nonempty
    T : fsu
    hT : HasSubset.Subset (ι T) C1
    ⊢ (Set.sInter (ι T)).Nonempty
  -/
  suffices ⋂₀ ι T ∈ C2 by exact claim6 _ this
  -- Finish
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    claim4 : FiniteInter (FiniteInter.finiteInterClosure C1)
    claim5 : FiniteInter C0
    claim6 : ∀ (P : Set (Ultrafilter X.A)), Membership.mem C2 P → P.Nonempty
    T : fsu
    hT : HasSubset.Subset (ι T) C1
    ⊢ Membership.mem C2 (Set.sInter (ι T))
  -/
  apply claim4.finiteInter_mem T
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    claim4 : FiniteInter (FiniteInter.finiteInterClosure C1)
    claim5 : FiniteInter C0
    claim6 : ∀ (P : Set (Ultrafilter X.A)), Membership.mem C2 P → P.Nonempty
    T : fsu
    hT : HasSubset.Subset (ι T) C1
    ⊢ HasSubset.Subset (↑T) (FiniteInter.finiteInterClosure C1)
  -/
  intro t ht
  /-
    case intro.intro
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem (Compactum.basic (Compactum.cl A)) F
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    C0 : ssu := setOf fun Z => Exists fun B => And (Membership.mem F B) (Eq (Set.p …
    AA : Set (Ultrafilter X.A) := setOf fun G => Membership.mem G A
    C1 : ssu := Insert.insert AA C0
    C2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure C1
    claim1 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → ∀ (C : Set (Ultr …
    claim2 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → B.Nonempty
    claim3 : ∀ (B : Set (Ultrafilter X.A)), Membership.mem C0 B → (Inter.inter AA  …
    claim4 : FiniteInter (FiniteInter.finiteInterClosure C1)
    claim5 : FiniteInter C0
    claim6 : ∀ (P : Set (Ultrafilter X.A)), Membership.mem C2 P → P.Nonempty
    T : fsu
    hT : HasSubset.Subset (ι T) C1
    t : Set (Ultrafilter X.A)
    ht : Membership.mem (↑T) t
    ⊢ Membership.mem (FiniteInter.finiteInterClosure C1) t
  -/
  exact finiteInterClosure.basic (@hT t ht)
  /-
    🎉 no goals
  -/


theorem isClosed_cl {X : Compactum} (A : Set X) : IsClosed (cl A) := by
  /-
    X : Compactum
    A : Set X.A
    ⊢ IsClosed (Compactum.cl A)
  -/
  rw [isClosed_iff]
  /-
    X : Compactum
    A : Set X.A
    ⊢ ∀ (F : Ultrafilter X.A), Membership.mem F (Compactum.cl A) → Membership.mem  …
  -/
  intro F hF
  /-
    X : Compactum
    A : Set X.A
    F : Ultrafilter X.A
    hF : Membership.mem F (Compactum.cl A)
    ⊢ Membership.mem (Compactum.cl A) (X.str F)
  -/
  exact cl_cl _ ⟨F, hF, rfl⟩
  /-
    🎉 no goals
  -/


theorem str_eq_of_le_nhds {X : Compactum} (F : Ultrafilter X) (x : X) : ↑F ≤ 𝓝 x → X.str F = x := by
  -- Notation to be used in this proof.
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let fsu := Finset (Set (Ultrafilter X))
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let ssu := Set (Set (Ultrafilter X))
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let ι : fsu → ssu := fun x ↦ ↑x
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let T0 : ssu := { S | ∃ A ∈ F, S = basic A }
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let AA := X.str ⁻¹' {x}
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let T1 := insert AA T0
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    T1 : ssu := Insert.insert AA T0
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  let T2 := finiteInterClosure T1
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    T1 : ssu := Insert.insert AA T0
    T2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure T1
    ⊢ LE.le (↑F) (nhds x) → Eq (X.str F) x
  -/
  intro cond
  -- If F contains a closed set A, then x is contained in A.
  have claim1 : ∀ A : Set X, IsClosed A → A ∈ F → x ∈ A := by
    intro A hA h
    by_contra H
    rw [le_nhds_iff] at cond
    specialize cond Aᶜ H hA.isOpen_compl
    rw [Ultrafilter.mem_coe, Ultrafilter.compl_mem_iff_not_mem] at cond
    contradiction
  -- If A ∈ F, then x ∈ cl A.
  have claim2 : ∀ A : Set X, A ∈ F → x ∈ cl A := by
    intro A hA
    exact claim1 (cl A) (isClosed_cl A) (mem_of_superset hA (subset_cl A))
  -- T0 is closed under intersections.
  have claim3 : ∀ (S1) (_ : S1 ∈ T0) (S2) (_ : S2 ∈ T0), S1 ∩ S2 ∈ T0 := by
    rintro S1 ⟨S1, hS1, rfl⟩ S2 ⟨S2, hS2, rfl⟩
    exact ⟨S1 ∩ S2, inter_mem hS1 hS2, by simp [basic_inter]⟩
  -- For every S ∈ T0, the intersection AA ∩ S is nonempty.
  have claim4 : ∀ S ∈ T0, (AA ∩ S).Nonempty := by
    rintro S ⟨S, hS, rfl⟩
    rcases claim2 _ hS with ⟨G, hG, hG2⟩
    exact ⟨G, hG2, hG⟩
  -- Every element of T0 is nonempty.
  have claim5 : ∀ S ∈ T0, Set.Nonempty S := by
    rintro S ⟨S, hS, rfl⟩
    exact ⟨F, hS⟩
  -- Every element of T2 is nonempty.
  have claim6 : ∀ S ∈ T2, Set.Nonempty S := by
    suffices ∀ S ∈ T2, S ∈ T0 ∨ ∃ Q ∈ T0, S = AA ∩ Q by
      intro S hS
      cases' this _ hS with h h
      · exact claim5 S h
      · rcases h with ⟨Q, hQ, rfl⟩
        exact claim4 Q hQ
    intro S hS
    apply finiteInterClosure_insert
    · constructor
      · use Set.univ
        refine ⟨Filter.univ_sets _, ?_⟩
        ext
        refine ⟨?_, by tauto⟩
        · intro
          apply Filter.univ_sets
      · exact claim3
    · exact hS
  -- It suffices to show that the intersection of any finite subset of T1 is nonempty.
  suffices ∀ F : fsu, ↑F ⊆ T1 → (⋂₀ ι F).Nonempty by
    obtain ⟨G, h1⟩ := Ultrafilter.exists_ultrafilter_of_finite_inter_nonempty _ this
    have c1 : X.join G = F := Ultrafilter.coe_le_coe.1 fun P hP => h1 (Or.inr ⟨P, hP, rfl⟩)
    have c2 : G.map X.str = X.incl x := by
      refine Ultrafilter.coe_le_coe.1 fun P hP => ?_
      apply mem_of_superset (h1 (Or.inl rfl))
      rintro x ⟨rfl⟩
      exact hP
    simp [← c1, c2]
  -- Finish...
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    T1 : ssu := Insert.insert AA T0
    T2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure T1
    cond : LE.le (↑F) (nhds x)
    claim1 : ∀ (A : Set X.A), IsClosed A → Membership.mem F A → Membership.mem A x
    claim2 : ∀ (A : Set X.A), Membership.mem F A → Membership.mem (Compactum.cl A) x
    claim3 : ∀ (S1 : Set (Ultrafilter X.A)), Membership.mem T0 S1 → ∀ (S2 : Set (U …
    claim4 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → (Inter.inter AA  …
    claim5 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → S.Nonempty
    claim6 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T2 S → S.Nonempty
    ⊢ ∀ (F : fsu), HasSubset.Subset (↑F) T1 → (Set.sInter (ι F)).Nonempty
  -/
  intro T hT
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    T1 : ssu := Insert.insert AA T0
    T2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure T1
    cond : LE.le (↑F) (nhds x)
    claim1 : ∀ (A : Set X.A), IsClosed A → Membership.mem F A → Membership.mem A x
    claim2 : ∀ (A : Set X.A), Membership.mem F A → Membership.mem (Compactum.cl A) x
    claim3 : ∀ (S1 : Set (Ultrafilter X.A)), Membership.mem T0 S1 → ∀ (S2 : Set (U …
    claim4 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → (Inter.inter AA  …
    claim5 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → S.Nonempty
    claim6 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T2 S → S.Nonempty
    T : fsu
    hT : HasSubset.Subset (↑T) T1
    ⊢ (Set.sInter (ι T)).Nonempty
  -/
  refine claim6 _ (finiteInter_mem (.finiteInterClosure_finiteInter _) _ ?_)
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    T1 : ssu := Insert.insert AA T0
    T2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure T1
    cond : LE.le (↑F) (nhds x)
    claim1 : ∀ (A : Set X.A), IsClosed A → Membership.mem F A → Membership.mem A x
    claim2 : ∀ (A : Set X.A), Membership.mem F A → Membership.mem (Compactum.cl A) x
    claim3 : ∀ (S1 : Set (Ultrafilter X.A)), Membership.mem T0 S1 → ∀ (S2 : Set (U …
    claim4 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → (Inter.inter AA  …
    claim5 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → S.Nonempty
    claim6 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T2 S → S.Nonempty
    T : fsu
    hT : HasSubset.Subset (↑T) T1
    ⊢ HasSubset.Subset (↑T) (FiniteInter.finiteInterClosure T1)
  -/
  intro t ht
  /-
    X : Compactum
    F : Ultrafilter X.A
    x : X.A
    fsu : Type u_1 := Finset (Set (Ultrafilter X.A))
    ssu : Type u_1 := Set (Set (Ultrafilter X.A))
    ι : fsu → ssu := fun x => ↑x
    T0 : ssu := setOf fun S => Exists fun A => And (Membership.mem F A) (Eq S (Com …
    AA : Set (Ultrafilter X.A) := Set.preimage X.str (Singleton.singleton x)
    T1 : ssu := Insert.insert AA T0
    T2 : Set (Set (Ultrafilter X.A)) := FiniteInter.finiteInterClosure T1
    cond : LE.le (↑F) (nhds x)
    claim1 : ∀ (A : Set X.A), IsClosed A → Membership.mem F A → Membership.mem A x
    claim2 : ∀ (A : Set X.A), Membership.mem F A → Membership.mem (Compactum.cl A) x
    claim3 : ∀ (S1 : Set (Ultrafilter X.A)), Membership.mem T0 S1 → ∀ (S2 : Set (U …
    claim4 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → (Inter.inter AA  …
    claim5 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T0 S → S.Nonempty
    claim6 : ∀ (S : Set (Ultrafilter X.A)), Membership.mem T2 S → S.Nonempty
    T : fsu
    hT : HasSubset.Subset (↑T) T1
    t : Set (Ultrafilter X.A)
    ht : Membership.mem (↑T) t
    ⊢ Membership.mem (FiniteInter.finiteInterClosure T1) t
  -/
  exact finiteInterClosure.basic (@hT t ht)
  /-
    🎉 no goals
  -/


theorem le_nhds_of_str_eq {X : Compactum} (F : Ultrafilter X) (x : X) : X.str F = x → ↑F ≤ 𝓝 x :=
                                                     /-
                                                       X : Compactum
                                                       F : Ultrafilter X.A
                                                       x : X.A
                                                       h : Eq (X.str F) x
                                                       s : Set X.A
                                                       hx : Membership.mem s x
                                                       hs : IsOpen s
                                                       ⊢ Membership.mem s (X.str F)
                                                     -/
  fun h => le_nhds_iff.mpr fun s hx hs => hs _ <| by rwa [h]
                                                     /-
                                                       🎉 no goals
                                                     -/

-- All the hard work above boils down to this `T2Space` instance.

instance {X : Compactum} : T2Space X := by
  /-
    X : Compactum
    ⊢ T2Space X.A
  -/
  rw [t2_iff_ultrafilter]
  /-
    X : Compactum
    ⊢ ∀ {x y : X.A} (f : Ultrafilter X.A), LE.le (↑f) (nhds x) → LE.le (↑f) (nhds  …
  -/
  intro _ _ F hx hy
  /-
    X : Compactum
    x✝ y✝ : X.A
    F : Ultrafilter X.A
    hx : LE.le (↑F) (nhds x✝)
    hy : LE.le (↑F) (nhds y✝)
    ⊢ Eq x✝ y✝
  -/
  rw [← str_eq_of_le_nhds _ _ hx, ← str_eq_of_le_nhds _ _ hy]
  /-
    🎉 no goals
  -/


/-- The structure map of a compactum actually computes limits. -/
theorem lim_eq_str {X : Compactum} (F : Ultrafilter X) : F.lim = X.str F := by
  /-
    X : Compactum
    F : Ultrafilter X.A
    ⊢ Eq F.lim (X.str F)
  -/
  rw [Ultrafilter.lim_eq_iff_le_nhds, le_nhds_iff]
  /-
    X : Compactum
    F : Ultrafilter X.A
    ⊢ ∀ (s : Set X.A), Membership.mem s (X.str F) → IsOpen s → Membership.mem (↑F) s
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem cl_eq_closure {X : Compactum} (A : Set X) : cl A = closure A := by
  /-
    X : Compactum
    A : Set X.A
    ⊢ Eq (Compactum.cl A) (closure A)
  -/
  ext
  /-
    case h
    X : Compactum
    A : Set X.A
    x✝ : X.A
    ⊢ Iff (Membership.mem (Compactum.cl A) x✝) (Membership.mem (closure A) x✝)
  -/
  rw [mem_closure_iff_ultrafilter]
  /-
    case h
    X : Compactum
    A : Set X.A
    x✝ : X.A
    ⊢ Iff (Membership.mem (Compactum.cl A) x✝) (Exists fun u => And (Membership.me …
  -/
  constructor
    /-
      case h.mp
      X : Compactum
      A : Set X.A
      x✝ : X.A
      ⊢ Membership.mem (Compactum.cl A) x✝ → Exists fun u => And (Membership.mem u A …
    -/
  · rintro ⟨F, h1, h2⟩
    /-
      case h.mp.intro.intro
      X : Compactum
      A : Set X.A
      x✝ : X.A
      F : Ultrafilter X.A
      h1 : Membership.mem (Compactum.basic A) F
      h2 : Eq (X.str F) x✝
      ⊢ Exists fun u => And (Membership.mem u A) (LE.le (↑u) (nhds x✝))
    -/
    exact ⟨F, h1, le_nhds_of_str_eq _ _ h2⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : Compactum
      A : Set X.A
      x✝ : X.A
      ⊢ (Exists fun u => And (Membership.mem u A) (LE.le (↑u) (nhds x✝))) → Membersh …
    -/
  · rintro ⟨F, h1, h2⟩
    /-
      case h.mpr.intro.intro
      X : Compactum
      A : Set X.A
      x✝ : X.A
      F : Ultrafilter X.A
      h1 : Membership.mem F A
      h2 : LE.le (↑F) (nhds x✝)
      ⊢ Membership.mem (Compactum.cl A) x✝
    -/
    exact ⟨F, h1, str_eq_of_le_nhds _ _ h2⟩
    /-
      🎉 no goals
    -/


/-- Any morphism of compacta is continuous. -/
theorem continuous_of_hom {X Y : Compactum} (f : X ⟶ Y) : Continuous f := by
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    ⊢ Continuous f.f
  -/
  rw [continuous_iff_ultrafilter]
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    ⊢ ∀ (x : X.A) (g : Ultrafilter X.A), LE.le (↑g) (nhds x) → Filter.Tendsto f.f  …
  -/
  intro x g h
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    x : X.A
    g : Ultrafilter X.A
    h : LE.le (↑g) (nhds x)
    ⊢ Filter.Tendsto f.f (↑g) (nhds (f.f x))
  -/
  rw [Tendsto, ← coe_map]
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    x : X.A
    g : Ultrafilter X.A
    h : LE.le (↑g) (nhds x)
    ⊢ LE.le (↑(Ultrafilter.map f.f g)) (nhds (f.f x))
  -/
  apply le_nhds_of_str_eq
  /-
    case a
    X Y : Compactum
    f : Quiver.Hom X Y
    x : X.A
    g : Ultrafilter X.A
    h : LE.le (↑g) (nhds x)
    ⊢ Eq (Y.str (Ultrafilter.map f.f g)) (f.f x)
  -/
  rw [← str_hom_commute, str_eq_of_le_nhds _ x _]
  /-
    X Y : Compactum
    f : Quiver.Hom X Y
    x : X.A
    g : Ultrafilter X.A
    h : LE.le (↑g) (nhds x)
    ⊢ LE.le (↑g) (nhds x)
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- Given any compact Hausdorff space, we construct a Compactum. -/
noncomputable def ofTopologicalSpace (X : Type*) [TopologicalSpace X] [CompactSpace X]
    [T2Space X] : Compactum where
  A := X
  a := Ultrafilter.lim
  unit := by
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
    -/
    ext x
    /-
      case h
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      x : (CategoryTheory.Functor.id (Type u_1)).obj X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
    -/
    exact lim_eq (pure_le_nhds _)
    /-
      🎉 no goals
    -/
  assoc := by
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
    -/
    ext FF
    /-
      case h
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      FF : ((CategoryTheory.ofTypeMonad Ultrafilter).comp (CategoryTheory.ofTypeMona …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
    -/
    change Ultrafilter (Ultrafilter X) at FF
    /-
      case h
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      FF : Ultrafilter (Ultrafilter X)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
    -/
    set x := (Ultrafilter.map Ultrafilter.lim FF).lim with c1
    have c2 : ∀ (U : Set X) (F : Ultrafilter X), F.lim ∈ U → IsOpen U → U ∈ F := by
      intro U F h1 hU
      exact isOpen_iff_ultrafilter.mp hU _ h1 _ (Ultrafilter.le_nhds_lim _)
    have c3 : ↑(Ultrafilter.map Ultrafilter.lim FF) ≤ 𝓝 x := by
      rw [le_nhds_iff]
      intro U hx hU
      exact mem_coe.2 (c2 _ _ (by rwa [← c1]) hU)
    have c4 : ∀ U : Set X, x ∈ U → IsOpen U → { G : Ultrafilter X | U ∈ G } ∈ FF := by
      intro U hx hU
      suffices Ultrafilter.lim ⁻¹' U ∈ FF by
        apply mem_of_superset this
        intro P hP
        exact c2 U P hP hU
      exact @c3 U (IsOpen.mem_nhds hU hx)
    /-
      case h
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      FF : Ultrafilter (Ultrafilter X)
      x : X := (Ultrafilter.map Ultrafilter.lim FF).lim
      c1 : Eq x (Ultrafilter.map Ultrafilter.lim FF).lim
      c2 : ∀ (U : Set X) (F : Ultrafilter X), Membership.mem U F.lim → IsOpen U → Me …
      c3 : LE.le (↑(Ultrafilter.map Ultrafilter.lim FF)) (nhds x)
      c4 : ∀ (U : Set X), Membership.mem U x → IsOpen U → Membership.mem FF (setOf f …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
    -/
    apply lim_eq
    /-
      case h.h
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      FF : Ultrafilter (Ultrafilter X)
      x : X := (Ultrafilter.map Ultrafilter.lim FF).lim
      c1 : Eq x (Ultrafilter.map Ultrafilter.lim FF).lim
      c2 : ∀ (U : Set X) (F : Ultrafilter X), Membership.mem U F.lim → IsOpen U → Me …
      c3 : LE.le (↑(Ultrafilter.map Ultrafilter.lim FF)) (nhds x)
      c4 : ∀ (U : Set X), Membership.mem U x → IsOpen U → Membership.mem FF (setOf f …
      ⊢ LE.le (↑((CategoryTheory.ofTypeMonad Ultrafilter).μ.app X FF)) (nhds (Catego …
    -/
    rw [le_nhds_iff]
    /-
      case h.h
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : CompactSpace X
      inst✝ : T2Space X
      FF : Ultrafilter (Ultrafilter X)
      x : X := (Ultrafilter.map Ultrafilter.lim FF).lim
      c1 : Eq x (Ultrafilter.map Ultrafilter.lim FF).lim
      c2 : ∀ (U : Set X) (F : Ultrafilter X), Membership.mem U F.lim → IsOpen U → Me …
      c3 : LE.le (↑(Ultrafilter.map Ultrafilter.lim FF)) (nhds x)
      c4 : ∀ (U : Set X), Membership.mem U x → IsOpen U → Membership.mem FF (setOf f …
      ⊢ ∀ (s : Set X), Membership.mem s (CategoryTheory.CategoryStruct.comp ((Catego …
    -/
    exact c4
    /-
      🎉 no goals
    -/


/-- Any continuous map between Compacta is a morphism of compacta. -/
def homOfContinuous {X Y : Compactum} (f : X → Y) (cont : Continuous f) : X ⟶ Y :=
  { f
    h := by
      /-
        X Y : Compactum
        f : X.A → Y.A
        cont : Continuous f
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
      -/
      rw [continuous_iff_ultrafilter] at cont
      /-
        X Y : Compactum
        f : X.A → Y.A
        cont : ∀ (x : X.A) (g : Ultrafilter X.A), LE.le (↑g) (nhds x) → Filter.Tendsto …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
      -/
      ext (F : Ultrafilter X)
      /-
        case h
        X Y : Compactum
        f : X.A → Y.A
        cont : ∀ (x : X.A) (g : Ultrafilter X.A), LE.le (↑g) (nhds x) → Filter.Tendsto …
        F : Ultrafilter X.A
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
      -/
      specialize cont (X.str F) F (le_nhds_of_str_eq F (X.str F) rfl)
      /-
        case h
        X Y : Compactum
        f : X.A → Y.A
        F : Ultrafilter X.A
        cont : Filter.Tendsto f (↑F) (nhds (f (X.str F)))
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ofTypeMonad Ultrafil …
      -/
      simp only [types_comp_apply, ofTypeFunctor_map]
      /-
        case h
        X Y : Compactum
        f : X.A → Y.A
        F : Ultrafilter X.A
        cont : Filter.Tendsto f (↑F) (nhds (f (X.str F)))
        ⊢ Eq (Y.a ((CategoryTheory.ofTypeMonad Ultrafilter).map f F)) (f (X.a F))
      -/
      exact str_eq_of_le_nhds (Ultrafilter.map f F) _ cont }
      /-
        🎉 no goals
      -/


/-- The functor functor from Compactum to CompHaus. -/
def compactumToCompHaus : Compactum ⥤ CompHaus where
  obj X := { toTop := { α := X }, prop := trivial }
  map := fun f =>
    { toFun := f
      continuous_toFun := Compactum.continuous_of_hom _ }


/-- The functor `compactumToCompHaus` is full. -/
instance full : compactumToCompHaus.{u}.Full where
  map_surjective f := ⟨Compactum.homOfContinuous f.1 f.2, rfl⟩


/-- The functor `compactumToCompHaus` is faithful. -/
instance faithful : compactumToCompHaus.Faithful where
  -- Porting note: this used to be obviously (though it consumed a bit of memory)
  map_injective := by
    /-
      ⊢ ∀ {X Y : Compactum}, Function.Injective compactumToCompHaus.map
    -/
    intro _ _ _ _ h
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` gets confused by coercion using forget.
    /-
      X✝ Y✝ : Compactum
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq (compactumToCompHaus.map a₁✝) (compactumToCompHaus.map a₂✝)
      ⊢ Eq a₁✝ a₂✝
    -/
    apply Monad.Algebra.Hom.ext
    /-
      case f
      X✝ Y✝ : Compactum
      a₁✝ a₂✝ : Quiver.Hom X✝ Y✝
      h : Eq (compactumToCompHaus.map a₁✝) (compactumToCompHaus.map a₂✝)
      ⊢ Eq a₁✝.f a₂✝.f
    -/
    apply congrArg (fun f => f.toFun) h
    /-
      🎉 no goals
    -/


/-- This definition is used to prove essential surjectivity of `compactumToCompHaus`. -/
def isoOfTopologicalSpace {D : CompHaus} :
    compactumToCompHaus.obj (Compactum.ofTopologicalSpace D) ≅ D where
  hom :=
    { toFun := id
      continuous_toFun :=
        continuous_def.2 fun _ h => by
          /-
            D : CompHaus
            x✝ : Set ↑D.toTop
            h : IsOpen x✝
            ⊢ IsOpen (Set.preimage id x✝)
          -/
          rw [isOpen_iff_ultrafilter'] at h
          /-
            D : CompHaus
            x✝ : Set ↑D.toTop
            h : ∀ (F : Ultrafilter ↑D.toTop), Membership.mem x✝ F.lim → Membership.mem (↑F …
            ⊢ IsOpen (Set.preimage id x✝)
          -/
          exact h }
          /-
            🎉 no goals
          -/
  inv :=
    { toFun := id
      continuous_toFun :=
        continuous_def.2 fun _ h1 => by
          /-
            D : CompHaus
            x✝ : Set ↑(compactumToCompHaus.obj (Compactum.ofTopologicalSpace ↑D.toTop)).to …
            h1 : IsOpen x✝
            ⊢ IsOpen (Set.preimage id x✝)
          -/
          rw [isOpen_iff_ultrafilter']
          /-
            D : CompHaus
            x✝ : Set ↑(compactumToCompHaus.obj (Compactum.ofTopologicalSpace ↑D.toTop)).to …
            h1 : IsOpen x✝
            ⊢ ∀ (F : Ultrafilter ↑D.toTop), Membership.mem (Set.preimage id x✝) F.lim → Me …
          -/
          intro _ h2
          /-
            D : CompHaus
            x✝ : Set ↑(compactumToCompHaus.obj (Compactum.ofTopologicalSpace ↑D.toTop)).to …
            h1 : IsOpen x✝
            F✝ : Ultrafilter ↑D.toTop
            h2 : Membership.mem (Set.preimage id x✝) F✝.lim
            ⊢ Membership.mem (↑F✝) (Set.preimage id x✝)
          -/
          exact h1 _ h2 }
          /-
            🎉 no goals
          -/


/-- The functor `compactumToCompHaus` is essentially surjective. -/
instance essSurj : compactumToCompHaus.EssSurj :=
  { mem_essImage := fun X => ⟨Compactum.ofTopologicalSpace X, ⟨isoOfTopologicalSpace⟩⟩ }


/-- The functor `compactumToCompHaus` is an equivalence of categories. -/
instance isEquivalence : compactumToCompHaus.IsEquivalence where


/-- The forgetful functors of `Compactum` and `CompHaus` are compatible via
`compactumToCompHaus`. -/
def compactumToCompHausCompForget :
    compactumToCompHaus ⋙ CategoryTheory.forget CompHaus ≅ Compactum.forget :=
  /-
    ⊢ ∀ {X Y : Compactum} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct. …
  -/
  NatIso.ofComponents fun _ => eqToIso rfl
  /-
    🎉 no goals
  -/

/-
TODO: `forget CompHaus` is monadic, as it is isomorphic to the composition
of an equivalence with the monadic functor `forget Compactum`.
Once we have the API to transfer monadicity of functors along such isomorphisms,
the instance `CreatesLimits (forget CompHaus)` can be deduced from this
monadicity.
-/

noncomputable instance CompHaus.forgetCreatesLimits : CreatesLimits (forget CompHaus) := by
  let e : forget CompHaus ≅ compactumToCompHaus.inv ⋙ Compactum.forget :=
    (((forget CompHaus).leftUnitor.symm ≪≫
    isoWhiskerRight compactumToCompHaus.asEquivalence.symm.unitIso (forget CompHaus)) ≪≫
    compactumToCompHaus.inv.associator compactumToCompHaus (forget CompHaus)) ≪≫
    isoWhiskerLeft _ compactumToCompHausCompForget
  /-
    e : CategoryTheory.Iso (CategoryTheory.forget CompHaus) (compactumToCompHaus.i …
    ⊢ CategoryTheory.CreatesLimits (CategoryTheory.forget CompHaus)
  -/
  exact createsLimitsOfNatIso e.symm
  /-
    🎉 no goals
  -/


noncomputable instance Profinite.forgetCreatesLimits : CreatesLimits (forget Profinite) := by
  /-
    ⊢ CategoryTheory.CreatesLimits (CategoryTheory.forget Profinite)
  -/
  change CreatesLimits (profiniteToCompHaus ⋙ forget _)
  /-
    ⊢ CategoryTheory.CreatesLimits (profiniteToCompHaus.comp (CategoryTheory.forge …
  -/
  infer_instance
  /-
    🎉 no goals
  -/

