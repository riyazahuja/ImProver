/--
The projection mapping everything that satisfies `J i` to itself, and everything else to `false`
-/
def Proj : (I → Bool) → (I → Bool) :=
  fun c i ↦ if J i then c i else false


@[simp]
theorem continuous_proj :
    Continuous (Proj J : (I → Bool) → (I → Bool)) := by
  /-
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    ⊢ Continuous (Profinite.NobelingProof.Proj J)
  -/
  dsimp (config := { unfoldPartialApp := true }) [Proj]
  /-
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    ⊢ Continuous fun c i => ite (J i) (c i) Bool.false
  -/
  apply continuous_pi
  /-
    case h
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    ⊢ ∀ (i : I), Continuous fun a => ite (J i) (a i) Bool.false
  -/
  intro i
  /-
    case h
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    i : I
    ⊢ Continuous fun a => ite (J i) (a i) Bool.false
  -/
  split
    /-
      case h.isTrue
      I : Type u
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      i : I
      h✝ : J i
      ⊢ Continuous fun a => a i
    -/
  · apply continuous_apply
    /-
      🎉 no goals
    -/
    /-
      case h.isFalse
      I : Type u
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      i : I
      h✝ : Not (J i)
      ⊢ Continuous fun a => Bool.false
    -/
  · apply continuous_const
    /-
      🎉 no goals
    -/


/-- The image of `Proj π J` -/
def π : Set (I → Bool) := (Proj J) '' C


/-- The restriction of `Proj π J` to a subset, mapping to its image. -/
@[simps!]
def ProjRestrict : C → π C J :=
  Set.MapsTo.restrict (Proj J) _ _ (Set.mapsTo_image _ _)


@[simp]
theorem continuous_projRestrict : Continuous (ProjRestrict C J) :=
  Continuous.restrict _ (continuous_proj _)


theorem proj_eq_self {x : I → Bool} (h : ∀ i, x i ≠ false → J i) : Proj J x = x := by
  /-
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    x : I → Bool
    h : ∀ (i : I), Ne (x i) Bool.false → J i
    ⊢ Eq (Profinite.NobelingProof.Proj J x) x
  -/
  ext i
  /-
    case h
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    x : I → Bool
    h : ∀ (i : I), Ne (x i) Bool.false → J i
    i : I
    ⊢ Eq (Profinite.NobelingProof.Proj J x i) (x i)
  -/
  simp only [Proj, ite_eq_left_iff]
  /-
    case h
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    x : I → Bool
    h : ∀ (i : I), Ne (x i) Bool.false → J i
    i : I
    ⊢ Not (J i) → Eq Bool.false (x i)
  -/
  contrapose!
  /-
    case h
    I : Type u
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    x : I → Bool
    h : ∀ (i : I), Ne (x i) Bool.false → J i
    i : I
    ⊢ Ne Bool.false (x i) → J i
  -/
  simpa only [ne_comm] using h i
  /-
    🎉 no goals
  -/


theorem proj_prop_eq_self (hh : ∀ i x, x ∈ C → x i ≠ false → J i) : π C J = C := by
  /-
    I : Type u
    C : Set (I → Bool)
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    hh : ∀ (i : I) (x : I → Bool), Membership.mem C x → Ne (x i) Bool.false → J i
    ⊢ Eq (Profinite.NobelingProof.π C J) C
  -/
  ext x
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    hh : ∀ (i : I) (x : I → Bool), Membership.mem C x → Ne (x i) Bool.false → J i
    x : I → Bool
    ⊢ Iff (Membership.mem (Profinite.NobelingProof.π C J) x) (Membership.mem C x)
  -/
  refine ⟨fun ⟨y, hy, h⟩ ↦ ?_, fun h ↦ ⟨x, h, ?_⟩⟩
    /-
      case h.refine_1
      I : Type u
      C : Set (I → Bool)
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      hh : ∀ (i : I) (x : I → Bool), Membership.mem C x → Ne (x i) Bool.false → J i
      x : I → Bool
      x✝ : Membership.mem (Profinite.NobelingProof.π C J) x
      y : I → Bool
      hy : Membership.mem C y
      h : Eq (Profinite.NobelingProof.Proj J y) x
      ⊢ Membership.mem C x
    -/
  · rwa [← h, proj_eq_self]; exact (hh · y hy)
                             /-
                               🎉 no goals
                             -/
    /-
      case h.refine_2
      I : Type u
      C : Set (I → Bool)
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      hh : ∀ (i : I) (x : I → Bool), Membership.mem C x → Ne (x i) Bool.false → J i
      x : I → Bool
      h : Membership.mem C x
      ⊢ Eq (Profinite.NobelingProof.Proj J x) x
    -/
  · rw [proj_eq_self]; exact (hh · x h)
                       /-
                         🎉 no goals
                       -/


theorem proj_comp_of_subset (h : ∀ i, J i → K i) : (Proj J ∘ Proj K) =
    (Proj J : (I → Bool) → (I → Bool)) := by
  /-
    I : Type u
    J K : I → Prop
    inst✝¹ : (i : I) → Decidable (J i)
    inst✝ : (i : I) → Decidable (K i)
    h : ∀ (i : I), J i → K i
    ⊢ Eq (Function.comp (Profinite.NobelingProof.Proj J) (Profinite.NobelingProof. …
  -/
  ext x i; dsimp [Proj]; aesop
                         /-
                           🎉 no goals
                         -/


theorem proj_eq_of_subset (h : ∀ i, J i → K i) : π (π C K) J = π C J := by
  /-
    I : Type u
    C : Set (I → Bool)
    J K : I → Prop
    inst✝¹ : (i : I) → Decidable (J i)
    inst✝ : (i : I) → Decidable (K i)
    h : ∀ (i : I), J i → K i
    ⊢ Eq (Profinite.NobelingProof.π (Profinite.NobelingProof.π C K) J) (Profinite. …
  -/
  ext x
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    J K : I → Prop
    inst✝¹ : (i : I) → Decidable (J i)
    inst✝ : (i : I) → Decidable (K i)
    h : ∀ (i : I), J i → K i
    x : I → Bool
    ⊢ Iff (Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.π C  …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case h.refine_1
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h✝ : ∀ (i : I), J i → K i
      x : I → Bool
      h : Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.π C K)  …
      ⊢ Membership.mem (Profinite.NobelingProof.π C J) x
    -/
  · obtain ⟨y, ⟨z, hz, rfl⟩, rfl⟩ := h
    /-
      case h.refine_1.intro.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h : ∀ (i : I), J i → K i
      z : I → Bool
      hz : Membership.mem C z
      ⊢ Membership.mem (Profinite.NobelingProof.π C J) (Profinite.NobelingProof.Proj …
    -/
    refine ⟨z, hz, (?_ : _ = (Proj J ∘ Proj K) z)⟩
    /-
      case h.refine_1.intro.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h : ∀ (i : I), J i → K i
      z : I → Bool
      hz : Membership.mem C z
      ⊢ Eq (Profinite.NobelingProof.Proj J z) (Function.comp (Profinite.NobelingProo …
    -/
    rw [proj_comp_of_subset J K h]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h✝ : ∀ (i : I), J i → K i
      x : I → Bool
      h : Membership.mem (Profinite.NobelingProof.π C J) x
      ⊢ Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.π C K) J) x
    -/
  · obtain ⟨y, hy, rfl⟩ := h
    /-
      case h.refine_2.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h : ∀ (i : I), J i → K i
      y : I → Bool
      hy : Membership.mem C y
      ⊢ Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.π C K) J) …
    -/
    dsimp [π]
    /-
      case h.refine_2.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h : ∀ (i : I), J i → K i
      y : I → Bool
      hy : Membership.mem C y
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Proj J) (Set.image (Profi …
    -/
    rw [← Set.image_comp]
    /-
      case h.refine_2.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h : ∀ (i : I), J i → K i
      y : I → Bool
      hy : Membership.mem C y
      ⊢ Membership.mem (Set.image (Function.comp (Profinite.NobelingProof.Proj J) (P …
    -/
    refine ⟨y, hy, ?_⟩
    /-
      case h.refine_2.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K : I → Prop
      inst✝¹ : (i : I) → Decidable (J i)
      inst✝ : (i : I) → Decidable (K i)
      h : ∀ (i : I), J i → K i
      y : I → Bool
      hy : Membership.mem C y
      ⊢ Eq (Function.comp (Profinite.NobelingProof.Proj J) (Profinite.NobelingProof. …
    -/
    rw [proj_comp_of_subset J K h]
    /-
      🎉 no goals
    -/


/-- A variant of `ProjRestrict` with domain of the form `π C K` -/
@[simps!]
def ProjRestricts (h : ∀ i, J i → K i) : π C K → π C J :=
  Homeomorph.setCongr (proj_eq_of_subset C J K h) ∘ ProjRestrict (π C K) J


@[simp]
theorem continuous_projRestricts (h : ∀ i, J i → K i) : Continuous (ProjRestricts C h) :=
  Continuous.comp (Homeomorph.continuous _) (continuous_projRestrict _ _)


theorem surjective_projRestricts (h : ∀ i, J i → K i) : Function.Surjective (ProjRestricts C h) :=
  (Homeomorph.surjective _).comp (Set.surjective_mapsTo_image_restrict _ _)


variable (J) in
theorem projRestricts_eq_id : ProjRestricts C (fun i (h : J i) ↦ h) = id := by
  /-
    I : Type u
    C : Set (I → Bool)
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    ⊢ Eq (Profinite.NobelingProof.ProjRestricts C ⋯) id
  -/
  ext ⟨x, y, hy, rfl⟩ i
  /-
    case h.mk.intro.intro.a.h
    I : Type u
    C : Set (I → Bool)
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    y : I → Bool
    hy : Membership.mem C y
    i : I
    ⊢ Eq (↑(Profinite.NobelingProof.ProjRestricts C ⋯ ⟨Profinite.NobelingProof.Pro …
  -/
  simp +contextual only [π, Proj, ProjRestricts_coe, id_eq, if_true]
  /-
    🎉 no goals
  -/


theorem projRestricts_eq_comp (hJK : ∀ i, J i → K i) (hKL : ∀ i, K i → L i) :
    ProjRestricts C hJK ∘ ProjRestricts C hKL = ProjRestricts C (fun i ↦ hKL i ∘ hJK i) := by
  /-
    I : Type u
    C : Set (I → Bool)
    J K L : I → Prop
    inst✝² : (i : I) → Decidable (J i)
    inst✝¹ : (i : I) → Decidable (K i)
    inst✝ : (i : I) → Decidable (L i)
    hJK : ∀ (i : I), J i → K i
    hKL : ∀ (i : I), K i → L i
    ⊢ Eq (Function.comp (Profinite.NobelingProof.ProjRestricts C hJK) (Profinite.N …
  -/
  ext x i
  /-
    case h.a.h
    I : Type u
    C : Set (I → Bool)
    J K L : I → Prop
    inst✝² : (i : I) → Decidable (J i)
    inst✝¹ : (i : I) → Decidable (K i)
    inst✝ : (i : I) → Decidable (L i)
    hJK : ∀ (i : I), J i → K i
    hKL : ∀ (i : I), K i → L i
    x : ↑(Profinite.NobelingProof.π C L)
    i : I
    ⊢ Eq (↑(Function.comp (Profinite.NobelingProof.ProjRestricts C hJK) (Profinite …
  -/
  simp only [π, Proj, Function.comp_apply, ProjRestricts_coe]
  /-
    case h.a.h
    I : Type u
    C : Set (I → Bool)
    J K L : I → Prop
    inst✝² : (i : I) → Decidable (J i)
    inst✝¹ : (i : I) → Decidable (K i)
    inst✝ : (i : I) → Decidable (L i)
    hJK : ∀ (i : I), J i → K i
    hKL : ∀ (i : I), K i → L i
    x : ↑(Profinite.NobelingProof.π C L)
    i : I
    ⊢ Eq (ite (J i) (ite (K i) (↑x i) Bool.false) Bool.false) (ite (J i) (↑x i) Bo …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem projRestricts_comp_projRestrict (h : ∀ i, J i → K i) :
    ProjRestricts C h ∘ ProjRestrict C K = ProjRestrict C J := by
  /-
    I : Type u
    C : Set (I → Bool)
    J K : I → Prop
    inst✝¹ : (i : I) → Decidable (J i)
    inst✝ : (i : I) → Decidable (K i)
    h : ∀ (i : I), J i → K i
    ⊢ Eq (Function.comp (Profinite.NobelingProof.ProjRestricts C h) (Profinite.Nob …
  -/
  ext x i
  /-
    case h.a.h
    I : Type u
    C : Set (I → Bool)
    J K : I → Prop
    inst✝¹ : (i : I) → Decidable (J i)
    inst✝ : (i : I) → Decidable (K i)
    h : ∀ (i : I), J i → K i
    x : ↑C
    i : I
    ⊢ Eq (↑(Function.comp (Profinite.NobelingProof.ProjRestricts C h) (Profinite.N …
  -/
  simp only [π, Proj, Function.comp_apply, ProjRestricts_coe, ProjRestrict_coe]
  /-
    case h.a.h
    I : Type u
    C : Set (I → Bool)
    J K : I → Prop
    inst✝¹ : (i : I) → Decidable (J i)
    inst✝ : (i : I) → Decidable (K i)
    h : ∀ (i : I), J i → K i
    x : ↑C
    i : I
    ⊢ Eq (ite (J i) (ite (K i) (↑x i) Bool.false) Bool.false) (ite (J i) (↑x i) Bo …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- The objectwise map in the isomorphism `spanFunctor ≅ Profinite.indexFunctor`. -/
def iso_map : C(π C J, (IndexFunctor.obj C J)) :=
  ⟨fun x ↦ ⟨fun i ↦ x.val i.val, by
    /-
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝² : (i : I) → Decidable (J i)
      inst✝¹ : (i : I) → Decidable (K i)
      inst✝ : (i : I) → Decidable (L i)
      x : ↑(Profinite.NobelingProof.π C J)
      ⊢ Membership.mem (Profinite.IndexFunctor.obj C J) fun i => ↑x ↑i
    -/
    rcases x with ⟨x, y, hy, rfl⟩
    /-
      case mk.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝² : (i : I) → Decidable (J i)
      inst✝¹ : (i : I) → Decidable (K i)
      inst✝ : (i : I) → Decidable (L i)
      y : I → Bool
      hy : Membership.mem C y
      ⊢ Membership.mem (Profinite.IndexFunctor.obj C J) fun i => ↑⟨Profinite.Nobelin …
    -/
    refine ⟨y, hy, ?_⟩
    /-
      case mk.intro.intro
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝² : (i : I) → Decidable (J i)
      inst✝¹ : (i : I) → Decidable (K i)
      inst✝ : (i : I) → Decidable (L i)
      y : I → Bool
      hy : Membership.mem C y
      ⊢ Eq ((ContinuousMap.precomp Subtype.val) y) fun i => ↑⟨Profinite.NobelingProo …
    -/
    ext ⟨i, hi⟩
    /-
      case mk.intro.intro.h.mk
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝² : (i : I) → Decidable (J i)
      inst✝¹ : (i : I) → Decidable (K i)
      inst✝ : (i : I) → Decidable (L i)
      y : I → Bool
      hy : Membership.mem C y
      i : I
      hi : J i
      ⊢ Eq ((ContinuousMap.precomp Subtype.val) y ⟨i, hi⟩) (↑⟨Profinite.NobelingProo …
    -/
    simp [precomp, Proj, hi]⟩, by
    /-
      🎉 no goals
    -/
    /-
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝² : (i : I) → Decidable (J i)
      inst✝¹ : (i : I) → Decidable (K i)
      inst✝ : (i : I) → Decidable (L i)
      ⊢ Continuous fun x => ⟨fun i => ↑x ↑i, ⋯⟩
    -/
    refine Continuous.subtype_mk (continuous_pi fun i ↦ ?_) _
    /-
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝² : (i : I) → Decidable (J i)
      inst✝¹ : (i : I) → Decidable (K i)
      inst✝ : (i : I) → Decidable (L i)
      i : Subtype fun i => J i
      ⊢ Continuous fun a => ↑a ↑i
    -/
    exact (continuous_apply i.val).comp continuous_subtype_val⟩
    /-
      🎉 no goals
    -/


lemma iso_map_bijective : Function.Bijective (iso_map C J) := by
  /-
    I : Type u
    C : Set (I → Bool)
    J : I → Prop
    inst✝ : (i : I) → Decidable (J i)
    ⊢ Function.Bijective ⇑(Profinite.NobelingProof.iso_map C J)
  -/
  refine ⟨fun a b h ↦ ?_, fun a ↦ ?_⟩
    /-
      case refine_1
      I : Type u
      C : Set (I → Bool)
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      a b : ↑(Profinite.NobelingProof.π C J)
      h : Eq ((Profinite.NobelingProof.iso_map C J) a) ((Profinite.NobelingProof.iso …
      ⊢ Eq a b
    -/
  · ext i
    /-
      case refine_1.a.h
      I : Type u
      C : Set (I → Bool)
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      a b : ↑(Profinite.NobelingProof.π C J)
      h : Eq ((Profinite.NobelingProof.iso_map C J) a) ((Profinite.NobelingProof.iso …
      i : I
      ⊢ Eq (↑a i) (↑b i)
    -/
    rw [Subtype.ext_iff] at h
    /-
      case refine_1.a.h
      I : Type u
      C : Set (I → Bool)
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      a b : ↑(Profinite.NobelingProof.π C J)
      h : Eq ↑((Profinite.NobelingProof.iso_map C J) a) ↑((Profinite.NobelingProof.i …
      i : I
      ⊢ Eq (↑a i) (↑b i)
    -/
    by_cases hi : J i
      /-
        case pos
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        a b : ↑(Profinite.NobelingProof.π C J)
        h : Eq ↑((Profinite.NobelingProof.iso_map C J) a) ↑((Profinite.NobelingProof.i …
        i : I
        hi : J i
        ⊢ Eq (↑a i) (↑b i)
      -/
    · exact congr_fun h ⟨i, hi⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        a b : ↑(Profinite.NobelingProof.π C J)
        h : Eq ↑((Profinite.NobelingProof.iso_map C J) a) ↑((Profinite.NobelingProof.i …
        i : I
        hi : Not (J i)
        ⊢ Eq (↑a i) (↑b i)
      -/
    · rcases a with ⟨_, c, hc, rfl⟩
      /-
        case neg.mk.intro.intro
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        b : ↑(Profinite.NobelingProof.π C J)
        i : I
        hi : Not (J i)
        c : I → Bool
        hc : Membership.mem C c
        h : Eq ↑((Profinite.NobelingProof.iso_map C J) ⟨Profinite.NobelingProof.Proj J …
        ⊢ Eq (↑⟨Profinite.NobelingProof.Proj J c, ⋯⟩ i) (↑b i)
      -/
      rcases b with ⟨_, d, hd, rfl⟩
      /-
        case neg.mk.intro.intro.mk.intro.intro
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        i : I
        hi : Not (J i)
        c : I → Bool
        hc : Membership.mem C c
        d : I → Bool
        hd : Membership.mem C d
        h : Eq ↑((Profinite.NobelingProof.iso_map C J) ⟨Profinite.NobelingProof.Proj J …
        ⊢ Eq (↑⟨Profinite.NobelingProof.Proj J c, ⋯⟩ i) (↑⟨Profinite.NobelingProof.Pro …
      -/
      simp only [Proj, if_neg hi]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      I : Type u
      C : Set (I → Bool)
      J : I → Prop
      inst✝ : (i : I) → Decidable (J i)
      a : ↑(Profinite.IndexFunctor.obj C J)
      ⊢ Exists fun a_1 => Eq ((Profinite.NobelingProof.iso_map C J) a_1) a
    -/
  · refine ⟨⟨fun i ↦ if hi : J i then a.val ⟨i, hi⟩ else false, ?_⟩, ?_⟩
      /-
        case refine_2.refine_1
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        a : ↑(Profinite.IndexFunctor.obj C J)
        ⊢ Membership.mem (Profinite.NobelingProof.π C J) fun i => dite (J i) (fun hi = …
      -/
    · rcases a with ⟨_, y, hy, rfl⟩
      /-
        case refine_2.refine_1.mk.intro.intro
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        y : (i : I) → (fun i => Bool) i
        hy : Membership.mem C y
        ⊢ Membership.mem (Profinite.NobelingProof.π C J) fun i => dite (J i) (fun hi = …
      -/
      exact ⟨y, hy, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        a : ↑(Profinite.IndexFunctor.obj C J)
        ⊢ Eq ((Profinite.NobelingProof.iso_map C J) ⟨fun i => dite (J i) (fun hi => ↑a …
      -/
    · ext i
      /-
        case refine_2.refine_2.a.h
        I : Type u
        C : Set (I → Bool)
        J : I → Prop
        inst✝ : (i : I) → Decidable (J i)
        a : ↑(Profinite.IndexFunctor.obj C J)
        i : Subtype fun i => J i
        ⊢ Eq (↑((Profinite.NobelingProof.iso_map C J) ⟨fun i => dite (J i) (fun hi =>  …
      -/
      exact dif_pos i.prop
      /-
        🎉 no goals
      -/


/--
For a given compact subset `C` of `I → Bool`, `spanFunctor` is the functor from the poset of finsets
of `I` to `Profinite`, sending a finite subset set `J` to the image of `C` under the projection
`Proj J`.
-/
noncomputable
def spanFunctor [∀ (s : Finset I) (i : I), Decidable (i ∈ s)] (hC : IsCompact C) :
    (Finset I)ᵒᵖ ⥤ Profinite.{u} where
  obj s := @Profinite.of (π C (· ∈ (unop s))) _
        /-
          I : Type u
          C : Set (I → Bool)
          J K L : I → Prop
          inst✝³ : (i : I) → Decidable (J i)
          inst✝² : (i : I) → Decidable (K i)
          inst✝¹ : (i : I) → Decidable (L i)
          inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
          hC : IsCompact C
          s : Opposite (Finset I)
          ⊢ CompactSpace ↑(Profinite.NobelingProof.π C fun x => Membership.mem (Opposite …
        -/
    (by rw [← isCompact_iff_compactSpace]; exact hC.image (continuous_proj _)) _ _
                                           /-
                                             🎉 no goals
                                           -/
  map h := ⟨(ProjRestricts C (leOfHom h.unop)), continuous_projRestricts _ _⟩
                 /-
                   I : Type u
                   C : Set (I → Bool)
                   J✝ K L : I → Prop
                   inst✝³ : (i : I) → Decidable (J✝ i)
                   inst✝² : (i : I) → Decidable (K i)
                   inst✝¹ : (i : I) → Decidable (L i)
                   inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
                   hC : IsCompact C
                   J : Opposite (Finset I)
                   ⊢ Eq ({ obj := fun s => Profinite.of ↑(Profinite.NobelingProof.π C fun x => Me …
                 -/
  map_id J := by simp only [projRestricts_eq_id C (· ∈ (unop J))]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                     /-
                       I : Type u
                       C : Set (I → Bool)
                       J K L : I → Prop
                       inst✝³ : (i : I) → Decidable (J i)
                       inst✝² : (i : I) → Decidable (K i)
                       inst✝¹ : (i : I) → Decidable (L i)
                       inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
                       hC : IsCompact C
                       X✝ Y✝ Z✝ : Opposite (Finset I)
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun s => Profinite.of ↑(Profinite.NobelingProof.π C fun x => Me …
                     -/
  map_comp _ _ := by dsimp; congr; dsimp; rw [projRestricts_eq_comp]
                                          /-
                                            🎉 no goals
                                          -/


/-- The limit cone on `spanFunctor` with point `C`. -/
noncomputable
def spanCone [∀ (s : Finset I) (i : I), Decidable (i ∈ s)] (hC : IsCompact C) :
    Cone (spanFunctor hC) where
                              /-
                                I : Type u
                                C : Set (I → Bool)
                                J K L : I → Prop
                                inst✝³ : (i : I) → Decidable (J i)
                                inst✝² : (i : I) → Decidable (K i)
                                inst✝¹ : (i : I) → Decidable (L i)
                                inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
                                hC : IsCompact C
                                ⊢ CompactSpace ↑C
                              -/
  pt := @Profinite.of C _ (by rwa [← isCompact_iff_compactSpace]) _ _
                              /-
                                🎉 no goals
                              -/
  π :=
  { app := fun s ↦ ⟨ProjRestrict C (· ∈ unop s), continuous_projRestrict _ _⟩
    naturality := by
      /-
        I : Type u
        C : Set (I → Bool)
        J K L : I → Prop
        inst✝³ : (i : I) → Decidable (J i)
        inst✝² : (i : I) → Decidable (K i)
        inst✝¹ : (i : I) → Decidable (L i)
        inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
        hC : IsCompact C
        ⊢ ∀ ⦃X Y : Opposite (Finset I)⦄ (f : Quiver.Hom X Y), Eq (CategoryTheory.Categ …
      -/
      intro X Y h
      simp only [Functor.const_obj_obj, Homeomorph.setCongr, Homeomorph.homeomorph_mk_coe,
        Functor.const_obj_map, Category.id_comp, ← projRestricts_comp_projRestrict C
        (leOfHom h.unop)]
      /-
        I : Type u
        C : Set (I → Bool)
        J K L : I → Prop
        inst✝³ : (i : I) → Decidable (J i)
        inst✝² : (i : I) → Decidable (K i)
        inst✝¹ : (i : I) → Decidable (L i)
        inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
        hC : IsCompact C
        X Y : Opposite (Finset I)
        h : Quiver.Hom X Y
        ⊢ Eq { toFun := Function.comp (Profinite.NobelingProof.ProjRestricts C ⋯) (Pro …
      -/
      rfl }
      /-
        🎉 no goals
      -/


/-- `spanCone` is a limit cone. -/
noncomputable
def spanCone_isLimit [∀ (s : Finset I) (i : I), Decidable (i ∈ s)] (hC : IsCompact C) :
    CategoryTheory.Limits.IsLimit (spanCone hC) := by
  refine (IsLimit.postcomposeHomEquiv (NatIso.ofComponents
    (fun s ↦ (CompHausLike.isoOfBijective _ (iso_map_bijective C (· ∈ unop s)))) ?_) (spanCone hC))
    (IsLimit.ofIsoLimit (indexCone_isLimit hC) (Cones.ext (Iso.refl _) ?_))
    /-
      case refine_1
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      ⊢ ∀ {X Y : Opposite (Finset I)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categ …
    -/
  · intro ⟨s⟩ ⟨t⟩ ⟨⟨⟨f⟩⟩⟩
    /-
      case refine_1
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      s t : Finset I
      f : LE.le (Opposite.unop { unop := t }) (Opposite.unop { unop := s })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Profinite.NobelingProof.spanFunctor …
    -/
    ext x
    have : iso_map C (· ∈ t) ∘ ProjRestricts C f = IndexFunctor.map C f ∘ iso_map C (· ∈ s) := by
      ext _ i; exact dif_pos i.prop
    /-
      case refine_1.w
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      s t : Finset I
      f : LE.le (Opposite.unop { unop := t }) (Opposite.unop { unop := s })
      x : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
      this : Eq (Function.comp (⇑(Profinite.NobelingProof.iso_map C fun x => Members …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((Profinite.NobelingProof.spanFuncto …
    -/
    exact congr_fun this x
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      ⊢ ∀ (j : Opposite (Finset I)), Eq ((Profinite.indexCone hC).π.app j) (Category …
    -/
  · intro ⟨s⟩
    /-
      case refine_2
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      s : Finset I
      ⊢ Eq ((Profinite.indexCone hC).π.app { unop := s }) (CategoryTheory.CategorySt …
    -/
    ext x
    have : iso_map C (· ∈ s) ∘ ProjRestrict C (· ∈ s) = IndexFunctor.π_app C (· ∈ s) := by
      ext _ i; exact dif_pos i.prop
    /-
      case refine_2.w
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      s : Finset I
      x : (CategoryTheory.forget Profinite).obj (((CategoryTheory.Functor.const (Opp …
      this : Eq (Function.comp (⇑(Profinite.NobelingProof.iso_map C fun x => Members …
      ⊢ Eq (((Profinite.indexCone hC).π.app { unop := s }) x) ((CategoryTheory.Categ …
    -/
    erw [← this]
    /-
      case refine_2.w
      I : Type u
      C : Set (I → Bool)
      J K L : I → Prop
      inst✝³ : (i : I) → Decidable (J i)
      inst✝² : (i : I) → Decidable (K i)
      inst✝¹ : (i : I) → Decidable (L i)
      inst✝ : (s : Finset I) → (i : I) → Decidable (Membership.mem s i)
      hC : IsCompact C
      s : Finset I
      x : (CategoryTheory.forget Profinite).obj (((CategoryTheory.Functor.const (Opp …
      this : Eq (Function.comp (⇑(Profinite.NobelingProof.iso_map C fun x => Members …
      ⊢ Eq (Function.comp (⇑(Profinite.NobelingProof.iso_map C fun x => Membership.m …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
`e C i` is the locally constant map from `C : Set (I → Bool)` to `ℤ` sending `f` to 1 if
`f.val i = true`, and 0 otherwise.
-/
def e (i : I) : LocallyConstant C ℤ where
  toFun := fun f ↦ (if f.val i then 1 else 0)
  isLocallyConstant := by
    /-
      I : Type u
      C : Set (I → Bool)
      i : I
      ⊢ IsLocallyConstant fun f => ite (Eq (↑f i) Bool.true) 1 0
    -/
    rw [IsLocallyConstant.iff_continuous]
    exact (continuous_of_discreteTopology (f := fun (a : Bool) ↦ (if a then (1 : ℤ) else 0))).comp
      ((continuous_apply i).comp continuous_subtype_val)


/--
`Products I` is the type of lists of decreasing elements of `I`, so a typical element is
`[i₁, i₂, ...]` with `i₁ > i₂ > ...`. We order `Products I` lexicographically, so `[] < [i₁, ...]`,
and `[i₁, i₂, ...] < [j₁, j₂, ...]` if either `i₁ < j₁`, or `i₁ = j₁` and `[i₂, ...] < [j₂, ...]`.

Terms `m = [i₁, i₂, ..., iᵣ]` of this type will be used to represent products of the form
`e C i₁ ··· e C iᵣ : LocallyConstant C ℤ` . The function associated to `m` is `m.eval`.
-/
def Products (I : Type*) [LinearOrder I] := {l : List I // l.Chain' (·>·)}


instance : LinearOrder (Products I) :=
  inferInstanceAs (LinearOrder {l : List I // l.Chain' (·>·)})


@[simp]
theorem lt_iff_lex_lt (l m : Products I) : l < m ↔ List.Lex (·<·) l.val m.val := by
  /-
    I : Type u
    inst✝ : LinearOrder I
    l m : Profinite.NobelingProof.Products I
    ⊢ Iff (LT.lt l m) (List.Lex (fun x1 x2 => LT.lt x1 x2) ↑l ↑m)
  -/
  cases l; cases m; rw [Subtype.mk_lt_mk]; exact Iff.rfl
                                           /-
                                             🎉 no goals
                                           -/


instance [WellFoundedLT I] : WellFoundedLT (Products I) := by
  have : (· < · : Products I → _ → _) = (fun l m ↦ List.Lex (·<·) l.val m.val) := by
    ext; exact lt_iff_lex_lt _ _
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    this : Eq (fun x1 x2 => LT.lt x1 x2) fun l m => List.Lex (fun x1 x2 => LT.lt x …
    ⊢ WellFoundedLT (Profinite.NobelingProof.Products I)
  -/
  rw [WellFoundedLT, this]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    this : Eq (fun x1 x2 => LT.lt x1 x2) fun l m => List.Lex (fun x1 x2 => LT.lt x …
    ⊢ IsWellFounded (Profinite.NobelingProof.Products I) fun l m => List.Lex (fun  …
  -/
  dsimp [Products]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    this : Eq (fun x1 x2 => LT.lt x1 x2) fun l m => List.Lex (fun x1 x2 => LT.lt x …
    ⊢ IsWellFounded (Subtype fun l => List.Chain' (fun x1 x2 => GT.gt x1 x2) l) fu …
  -/
  rw [(by rfl : (·>· : I → _) = flip (·<·))]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    this : Eq (fun x1 x2 => LT.lt x1 x2) fun l m => List.Lex (fun x1 x2 => LT.lt x …
    ⊢ IsWellFounded (Subtype fun l => List.Chain' (flip fun x1 x2 => LT.lt x1 x2)  …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The evaluation `e C i₁ ··· e C iᵣ : C → ℤ`  of a formal product `[i₁, i₂, ..., iᵣ]`. -/
def eval (l : Products I) := (l.1.map (e C)).prod


/--
The predicate on products which we prove picks out a basis of `LocallyConstant C ℤ`. We call such a
product "good".
-/
def isGood (l : Products I) : Prop :=
  l.eval C ∉ Submodule.span ℤ ((Products.eval C) '' {m | m < l})


theorem rel_head!_of_mem [Inhabited I] {i : I} {l : Products I} (hi : i ∈ l.val) :
    i ≤ l.val.head! :=
  List.Sorted.le_head! (List.chain'_iff_pairwise.mp l.prop) hi


theorem head!_le_of_lt [Inhabited I] {q l : Products I} (h : q < l) (hq : q.val ≠ []) :
    q.val.head! ≤ l.val.head! :=
  List.head!_le_of_lt l.val q.val h hq


/-- The set of good products. -/
def GoodProducts := {l : Products I | l.isGood C}


/-- Evaluation of good products. -/
def eval (l : {l : Products I // l.isGood C}) : LocallyConstant C ℤ :=
  Products.eval C l.1


theorem injective : Function.Injective (eval C) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    ⊢ Function.Injective (Profinite.NobelingProof.GoodProducts.eval C)
  -/
  intro ⟨a, ha⟩ ⟨b, hb⟩ h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    a : Profinite.NobelingProof.Products I
    ha : Profinite.NobelingProof.Products.isGood C a
    b : Profinite.NobelingProof.Products I
    hb : Profinite.NobelingProof.Products.isGood C b
    h : Eq (Profinite.NobelingProof.GoodProducts.eval C ⟨a, ha⟩) (Profinite.Nobeli …
    ⊢ Eq ⟨a, ha⟩ ⟨b, hb⟩
  -/
  dsimp [eval] at h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    a : Profinite.NobelingProof.Products I
    ha : Profinite.NobelingProof.Products.isGood C a
    b : Profinite.NobelingProof.Products I
    hb : Profinite.NobelingProof.Products.isGood C b
    h : Eq (Profinite.NobelingProof.Products.eval C a) (Profinite.NobelingProof.Pr …
    ⊢ Eq ⟨a, ha⟩ ⟨b, hb⟩
  -/
  rcases lt_trichotomy a b with (h'|rfl|h')
    /-
      case inl
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      a : Profinite.NobelingProof.Products I
      ha : Profinite.NobelingProof.Products.isGood C a
      b : Profinite.NobelingProof.Products I
      hb : Profinite.NobelingProof.Products.isGood C b
      h : Eq (Profinite.NobelingProof.Products.eval C a) (Profinite.NobelingProof.Pr …
      h' : LT.lt a b
      ⊢ Eq ⟨a, ha⟩ ⟨b, hb⟩
    -/
  · exfalso; apply hb; rw [← h]
    /-
      case inl
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      a : Profinite.NobelingProof.Products I
      ha : Profinite.NobelingProof.Products.isGood C a
      b : Profinite.NobelingProof.Products I
      hb : Profinite.NobelingProof.Products.isGood C b
      h : Eq (Profinite.NobelingProof.Products.eval C a) (Profinite.NobelingProof.Pr …
      h' : LT.lt a b
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    exact Submodule.subset_span ⟨a, h', rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      a : Profinite.NobelingProof.Products I
      ha hb : Profinite.NobelingProof.Products.isGood C a
      h : Eq (Profinite.NobelingProof.Products.eval C a) (Profinite.NobelingProof.Pr …
      ⊢ Eq ⟨a, ha⟩ ⟨a, hb⟩
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      a : Profinite.NobelingProof.Products I
      ha : Profinite.NobelingProof.Products.isGood C a
      b : Profinite.NobelingProof.Products I
      hb : Profinite.NobelingProof.Products.isGood C b
      h : Eq (Profinite.NobelingProof.Products.eval C a) (Profinite.NobelingProof.Pr …
      h' : LT.lt b a
      ⊢ Eq ⟨a, ha⟩ ⟨b, hb⟩
    -/
  · exfalso; apply ha; rw [h]
    /-
      case inr.inr
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      a : Profinite.NobelingProof.Products I
      ha : Profinite.NobelingProof.Products.isGood C a
      b : Profinite.NobelingProof.Products I
      hb : Profinite.NobelingProof.Products.isGood C b
      h : Eq (Profinite.NobelingProof.Products.eval C a) (Profinite.NobelingProof.Pr …
      h' : LT.lt b a
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    exact Submodule.subset_span ⟨b, ⟨h',rfl⟩⟩
    /-
      🎉 no goals
    -/


/-- The image of the good products in the module `LocallyConstant C ℤ`. -/
def range := Set.range (GoodProducts.eval C)


/-- The type of good products is equivalent to its image. -/
noncomputable
def equiv_range : GoodProducts C ≃ range C :=
  Equiv.ofInjective (eval C) (injective C)


theorem equiv_toFun_eq_eval : (equiv_range C).toFun = Set.rangeFactorization (eval C) := rfl


theorem linearIndependent_iff_range : LinearIndependent ℤ (GoodProducts.eval C) ↔
    LinearIndependent ℤ (fun (p : range C) ↦ p.1) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    ⊢ Iff (LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)) (L …
  -/
  rw [← @Set.rangeFactorization_eq _ _ (GoodProducts.eval C), ← equiv_toFun_eq_eval C]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    ⊢ Iff (LinearIndependent Int (Function.comp Subtype.val (Profinite.NobelingPro …
  -/
  exact linearIndependent_equiv (equiv_range C)
  /-
    🎉 no goals
  -/


theorem eval_eq (l : Products I) (x : C) :
    l.eval C x = if ∀ i, i ∈ l.val → (x.val i = true) then 1 else 0 := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    x : ↑C
    ⊢ Eq ((Profinite.NobelingProof.Products.eval C l) x) (ite (∀ (i : I), Membersh …
  -/
  change LocallyConstant.evalMonoidHom x (l.eval C) = _
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    x : ↑C
    ⊢ Eq ((LocallyConstant.evalMonoidHom x) (Profinite.NobelingProof.Products.eval …
  -/
  rw [eval, map_list_prod]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    x : ↑C
    ⊢ Eq (List.map (⇑(LocallyConstant.evalMonoidHom x)) (List.map (Profinite.Nobel …
  -/
  split_ifs with h
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : ∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true
      ⊢ Eq (List.map (⇑(LocallyConstant.evalMonoidHom x)) (List.map (Profinite.Nobel …
    -/
  · simp only [List.map_map]
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : ∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true
      ⊢ Eq (List.map (Function.comp (⇑(LocallyConstant.evalMonoidHom x)) (Profinite. …
    -/
    apply List.prod_eq_one
    /-
      case pos.hl
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : ∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true
      ⊢ ∀ (x_1 : Int), Membership.mem (List.map (Function.comp (⇑(LocallyConstant.ev …
    -/
    simp only [List.mem_map, Function.comp_apply]
    /-
      case pos.hl
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : ∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true
      ⊢ ∀ (x_1 : Int), (Exists fun a => And (Membership.mem (↑l) a) (Eq ((LocallyCon …
    -/
    rintro _ ⟨i, hi, rfl⟩
    /-
      case pos.hl.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : ∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true
      i : I
      hi : Membership.mem (↑l) i
      ⊢ Eq ((LocallyConstant.evalMonoidHom x) (Profinite.NobelingProof.e C i)) 1
    -/
    exact if_pos (h i hi)
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : Not (∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true)
      ⊢ Eq (List.map (⇑(LocallyConstant.evalMonoidHom x)) (List.map (Profinite.Nobel …
    -/
  · simp only [List.map_map, List.prod_eq_zero_iff, List.mem_map, Function.comp_apply]
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : Not (∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true)
      ⊢ Exists fun a => And (Membership.mem (↑l) a) (Eq ((LocallyConstant.evalMonoid …
    -/
    push_neg at h
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : Exists fun i => And (Membership.mem (↑l) i) (Ne (↑x i) Bool.true)
      ⊢ Exists fun a => And (Membership.mem (↑l) a) (Eq ((LocallyConstant.evalMonoid …
    -/
    convert h with i
    /-
      case h.e'_2.h.h.e'_2.a
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : Exists fun i => And (Membership.mem (↑l) i) (Ne (↑x i) Bool.true)
      i : I
      ⊢ Iff (Eq ((LocallyConstant.evalMonoidHom x) (Profinite.NobelingProof.e C i))  …
    -/
    dsimp [LocallyConstant.evalMonoidHom, e]
    /-
      case h.e'_2.h.h.e'_2.a
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      l : Profinite.NobelingProof.Products I
      x : ↑C
      h : Exists fun i => And (Membership.mem (↑l) i) (Ne (↑x i) Bool.true)
      i : I
      ⊢ Iff (Eq (ite (Eq (↑x i) Bool.true) 1 0) 0) (Not (Eq (↑x i) Bool.true))
    -/
    simp only [ite_eq_right_iff, one_ne_zero]
    /-
      🎉 no goals
    -/


theorem evalFacProp {l : Products I} (J : I → Prop)
    (h : ∀ a, a ∈ l.val → J a) [∀ j, Decidable (J j)] :
    l.eval (π C J) ∘ ProjRestrict C J = l.eval C := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    ⊢ Eq (Function.comp (⇑(Profinite.NobelingProof.Products.eval (Profinite.Nobeli …
  -/
  ext x
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    x : ↑C
    ⊢ Eq (Function.comp (⇑(Profinite.NobelingProof.Products.eval (Profinite.Nobeli …
  -/
  dsimp [ProjRestrict]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    x : ↑C
    ⊢ Eq ((Profinite.NobelingProof.Products.eval (Profinite.NobelingProof.π C J) l …
  -/
  rw [Products.eval_eq, Products.eval_eq]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    x : ↑C
    ⊢ Eq (ite (∀ (i : I), Membership.mem (↑l) i → Eq (↑(Set.MapsTo.restrict (Profi …
  -/
  congr
  /-
    case h.e_c
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    x : ↑C
    ⊢ Eq (∀ (i : I), Membership.mem (↑l) i → Eq (↑(Set.MapsTo.restrict (Profinite. …
  -/
  apply forall_congr; intro i
  /-
    case h.e_c.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    x : ↑C
    i : I
    ⊢ Eq (Membership.mem (↑l) i → Eq (↑(Set.MapsTo.restrict (Profinite.NobelingPro …
  -/
  apply forall_congr; intro hi
  /-
    case h.e_c.h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝ : (j : I) → Decidable (J j)
    x : ↑C
    i : I
    hi : Membership.mem (↑l) i
    ⊢ Eq (Eq (↑(Set.MapsTo.restrict (Profinite.NobelingProof.Proj J) C (Profinite. …
  -/
  simp [h i hi, Proj]
  /-
    🎉 no goals
  -/


theorem evalFacProps {l : Products I} (J K : I → Prop)
    (h : ∀ a, a ∈ l.val → J a) [∀ j, Decidable (J j)] [∀ j, Decidable (K j)]
    (hJK : ∀ i, J i → K i) :
    l.eval (π C J) ∘ ProjRestricts C hJK = l.eval (π C K) := by
  have : l.eval (π C J) ∘ Homeomorph.setCongr (proj_eq_of_subset C J K hJK) =
      l.eval (π (π C K) J) := by
    ext; simp [Homeomorph.setCongr, Products.eval_eq]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J K : I → Prop
    h : ∀ (a : I), Membership.mem (↑l) a → J a
    inst✝¹ : (j : I) → Decidable (J j)
    inst✝ : (j : I) → Decidable (K j)
    hJK : ∀ (i : I), J i → K i
    this : Eq (Function.comp ⇑(Profinite.NobelingProof.Products.eval (Profinite.No …
    ⊢ Eq (Function.comp (⇑(Profinite.NobelingProof.Products.eval (Profinite.Nobeli …
  -/
  rw [ProjRestricts, ← Function.comp_assoc, this, ← evalFacProp (π C K) J h]
  /-
    🎉 no goals
  -/


theorem prop_of_isGood {l : Products I} (J : I → Prop) [∀ j, Decidable (J j)]
    (h : l.isGood (π C J)) : ∀ a, a ∈ l.val → J a := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    inst✝ : (j : I) → Decidable (J j)
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C J) l
    ⊢ ∀ (a : I), Membership.mem (↑l) a → J a
  -/
  intro i hi
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    inst✝ : (j : I) → Decidable (J j)
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C J) l
    i : I
    hi : Membership.mem (↑l) i
    ⊢ J i
  -/
  by_contra h'
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    inst✝ : (j : I) → Decidable (J j)
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C J) l
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (J i)
    ⊢ False
  -/
  apply h
  suffices eval (π C J) l = 0 by
    rw [this]
    exact Submodule.zero_mem _
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    inst✝ : (j : I) → Decidable (J j)
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C J) l
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (J i)
    ⊢ Eq (Profinite.NobelingProof.Products.eval (Profinite.NobelingProof.π C J) l) 0
  -/
  ext ⟨_, _, _, rfl⟩
  /-
    case h.mk.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    inst✝ : (j : I) → Decidable (J j)
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C J) l
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (J i)
    w✝ : I → Bool
    left✝ : Membership.mem C w✝
    ⊢ Eq ((Profinite.NobelingProof.Products.eval (Profinite.NobelingProof.π C J) l …
  -/
  rw [eval_eq, if_neg fun h ↦ ?_, LocallyConstant.zero_apply]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    l : Profinite.NobelingProof.Products I
    J : I → Prop
    inst✝ : (j : I) → Decidable (J j)
    h✝ : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C J) l
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (J i)
    w✝ : I → Bool
    left✝ : Membership.mem C w✝
    h : ∀ (i : I), Membership.mem (↑l) i → Eq (↑⟨Profinite.NobelingProof.Proj J w✝ …
    ⊢ False
  -/
  simpa [Proj, h'] using h i hi
  /-
    🎉 no goals
  -/


/-- The good products span `LocallyConstant C ℤ` if and only all the products do. -/
theorem GoodProducts.span_iff_products [WellFoundedLT I] :
    ⊤ ≤ Submodule.span ℤ (Set.range (eval C)) ↔
      ⊤ ≤ Submodule.span ℤ (Set.range (Products.eval C)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    ⊢ Iff (LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.G …
  -/
  refine ⟨fun h ↦ le_trans h (span_mono (fun a ⟨b, hb⟩ ↦ ⟨b.val, hb⟩)), fun h ↦ le_trans h ?_⟩
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Prod …
    ⊢ LE.le (Submodule.span Int (Set.range (Profinite.NobelingProof.Products.eval  …
  -/
  rw [span_le]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Prod …
    ⊢ HasSubset.Subset (Set.range (Profinite.NobelingProof.Products.eval C)) ↑(Sub …
  -/
  rintro f ⟨l, rfl⟩
  /-
    case intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Prod …
    l : Profinite.NobelingProof.Products I
    ⊢ Membership.mem (↑(Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
  -/
  let L : Products I → Prop := fun m ↦ m.eval C ∈ span ℤ (Set.range (GoodProducts.eval C))
  /-
    case intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Prod …
    l : Profinite.NobelingProof.Products I
    L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
    ⊢ Membership.mem (↑(Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
  -/
  suffices L l by assumption
  /-
    case intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Prod …
    l : Profinite.NobelingProof.Products I
    L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
    ⊢ L l
  -/
  apply IsWellFounded.induction (·<· : Products I → Products I → Prop)
  /-
    case intro.ind
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Prod …
    l : Profinite.NobelingProof.Products I
    L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
    ⊢ ∀ (x : Profinite.NobelingProof.Products I), (∀ (y : Profinite.NobelingProof. …
  -/
  intro l h
  /-
    case intro.ind
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
    l✝ : Profinite.NobelingProof.Products I
    L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
    l : Profinite.NobelingProof.Products I
    h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
    ⊢ L l
  -/
  dsimp
  /-
    case intro.ind
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
    l✝ : Profinite.NobelingProof.Products I
    L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
    l : Profinite.NobelingProof.Products I
    h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
    ⊢ L l
  -/
  by_cases hl : l.isGood C
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
      l✝ : Profinite.NobelingProof.Products I
      L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
      l : Profinite.NobelingProof.Products I
      h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
      hl : Profinite.NobelingProof.Products.isGood C l
      ⊢ L l
    -/
  · apply subset_span
    /-
      case pos.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
      l✝ : Profinite.NobelingProof.Products I
      L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
      l : Profinite.NobelingProof.Products I
      h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
      hl : Profinite.NobelingProof.Products.isGood C l
      ⊢ Membership.mem (Set.range (Profinite.NobelingProof.GoodProducts.eval C)) (Pr …
    -/
    exact ⟨⟨l, hl⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
      l✝ : Profinite.NobelingProof.Products I
      L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
      l : Profinite.NobelingProof.Products I
      h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
      hl : Not (Profinite.NobelingProof.Products.isGood C l)
      ⊢ L l
    -/
  · simp only [Products.isGood, not_not] at hl
    suffices Products.eval C '' {m | m < l} ⊆ span ℤ (Set.range (GoodProducts.eval C)) by
      rw [← span_le] at this
      exact this hl
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
      l✝ : Profinite.NobelingProof.Products I
      L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
      l : Profinite.NobelingProof.Products I
      h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
      hl : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pr …
      ⊢ HasSubset.Subset (Set.image (Profinite.NobelingProof.Products.eval C) (setOf …
    -/
    rintro a ⟨m, hm, rfl⟩
    /-
      case neg.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      h✝ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
      l✝ : Profinite.NobelingProof.Products I
      L : Profinite.NobelingProof.Products I → Prop := fun m => Membership.mem (Subm …
      l : Profinite.NobelingProof.Products I
      h : ∀ (y : Profinite.NobelingProof.Products I), LT.lt y l → L y
      hl : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pr …
      m : Profinite.NobelingProof.Products I
      hm : Membership.mem (setOf fun m => LT.lt m l) m
      ⊢ Membership.mem (↑(Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    -/
    exact h m hm
    /-
      🎉 no goals
    -/


/-- The `ℤ`-linear map induced by precomposition of the projection `C → π C (· ∈ s)`. -/
noncomputable
def πJ : LocallyConstant (π C (· ∈ s)) ℤ →ₗ[ℤ] LocallyConstant C ℤ :=
  LocallyConstant.comapₗ ℤ ⟨_, (continuous_projRestrict C (· ∈ s))⟩


theorem eval_eq_πJ (l : Products I) (hl : l.isGood (π C (· ∈ s))) :
    l.eval C = πJ C s (l.eval (π C (· ∈ s))) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    l : Profinite.NobelingProof.Products I
    hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
    ⊢ Eq (Profinite.NobelingProof.Products.eval C l) ((Profinite.NobelingProof.πJ  …
  -/
  ext f
  simp only [πJ, LocallyConstant.comapₗ, LinearMap.coe_mk, AddHom.coe_mk,
    (continuous_projRestrict C (· ∈ s)), LocallyConstant.coe_comap, Function.comp_apply]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    l : Profinite.NobelingProof.Products I
    hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
    f : ↑C
    ⊢ Eq ((Profinite.NobelingProof.Products.eval C l) f) ((Profinite.NobelingProof …
  -/
  exact (congr_fun (Products.evalFacProp C (· ∈ s) (Products.prop_of_isGood  C (· ∈ s) hl)) _).symm
  /-
    🎉 no goals
  -/


/-- `π C (· ∈ s)` is finite for a finite set `s`. -/
noncomputable
instance : Fintype (π C (· ∈ s)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    ⊢ Fintype ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
  -/
  let f : π C (· ∈ s) → (s → Bool) := fun x j ↦ x.val j.val
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x) → (Subtype fun  …
    ⊢ Fintype ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
  -/
  refine Fintype.ofInjective f ?_
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x) → (Subtype fun  …
    ⊢ Function.Injective f
  -/
  intro ⟨_, x, hx, rfl⟩ ⟨_, y, hy, rfl⟩ h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x) → (Subtype fun  …
    x : I → Bool
    hx : Membership.mem C x
    y : I → Bool
    hy : Membership.mem C y
    h : Eq (f ⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩) ( …
    ⊢ Eq ⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩ ⟨Profin …
  -/
  ext i
  /-
    case a.h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x) → (Subtype fun  …
    x : I → Bool
    hx : Membership.mem C x
    y : I → Bool
    hy : Membership.mem C y
    h : Eq (f ⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩) ( …
    i : I
    ⊢ Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩ i) (↑ …
  -/
  by_cases hi : i ∈ s
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      f : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x) → (Subtype fun  …
      x : I → Bool
      hx : Membership.mem C x
      y : I → Bool
      hy : Membership.mem C y
      h : Eq (f ⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩) ( …
      i : I
      hi : Membership.mem s i
      ⊢ Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩ i) (↑ …
    -/
  · exact congrFun h ⟨i, hi⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      f : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x) → (Subtype fun  …
      x : I → Bool
      hx : Membership.mem C x
      y : I → Bool
      hy : Membership.mem C y
      h : Eq (f ⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩) ( …
      i : I
      hi : Not (Membership.mem s i)
      ⊢ Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) x, ⋯⟩ i) (↑ …
    -/
  · simp only [Proj, if_neg hi]
    /-
      🎉 no goals
    -/



open scoped Classical in
/-- The Kronecker delta as a locally constant map from `π C (· ∈ s)` to `ℤ`. -/
noncomputable
def spanFinBasis (x : π C (· ∈ s)) : LocallyConstant (π C (· ∈ s)) ℤ where
  toFun := fun y ↦ if y = x then 1 else 0
  isLocallyConstant :=
    haveI : DiscreteTopology (π C (· ∈ s)) := Finite.instDiscreteTopology
    IsLocallyConstant.of_discrete _


open scoped Classical in
theorem spanFinBasis.span : ⊤ ≤ Submodule.span ℤ (Set.range (spanFinBasis C s)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    ⊢ LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.spanFi …
  -/
  intro f _
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem s x …
    a✝ : Membership.mem Top.top f
    ⊢ Membership.mem (Submodule.span Int (Set.range (Profinite.NobelingProof.spanF …
  -/
  rw [Finsupp.mem_span_range_iff_exists_finsupp]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem s x …
    a✝ : Membership.mem Top.top f
    ⊢ Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof. …
  -/
  use Finsupp.onFinset (Finset.univ) f.toFun (fun _ _ ↦ Finset.mem_univ _)
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem s x …
    a✝ : Membership.mem Top.top f
    ⊢ Eq ((Finsupp.onFinset Finset.univ f.toFun ⋯).sum fun i a => HSMul.hSMul a (P …
  -/
  ext x
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem s x …
    a✝ : Membership.mem Top.top f
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    ⊢ Eq (((Finsupp.onFinset Finset.univ f.toFun ⋯).sum fun i a => HSMul.hSMul a ( …
  -/
  change LocallyConstant.evalₗ ℤ x _ = _
  simp only [zsmul_eq_mul, map_finsupp_sum, LocallyConstant.evalₗ_apply,
    LocallyConstant.coe_mul, Pi.mul_apply, spanFinBasis, LocallyConstant.coe_mk, mul_ite, mul_one,
    mul_zero, Finsupp.sum_ite_eq, Finsupp.mem_support_iff, ne_eq, ite_not]
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem s x …
    a✝ : Membership.mem Top.top f
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    ⊢ Eq (ite (Eq ((Finsupp.onFinset Finset.univ f.toFun ⋯) x) 0) 0 (↑((Finsupp.on …
  -/
  split_ifs with h <;> [exact h.symm; rfl]
  /-
    🎉 no goals
  -/


/--
A certain explicit list of locally constant maps. The theorem `factors_prod_eq_basis` shows that the
product of the elements in this list is the delta function `spanFinBasis C s x`.
-/
def factors (x : π C (· ∈ s)) : List (LocallyConstant (π C (· ∈ s)) ℤ) :=
  List.map (fun i ↦ if x.val i = true then e (π C (· ∈ s)) i else (1 - (e (π C (· ∈ s)) i)))
    (s.sort (·≥·))


theorem list_prod_apply {I} (C : Set (I → Bool)) (x : C) (l : List (LocallyConstant C ℤ)) :
    l.prod x = (l.map (LocallyConstant.evalMonoidHom x)).prod := by
  /-
    I : Type u_1
    C : Set (I → Bool)
    x : ↑C
    l : List (LocallyConstant (↑C) Int)
    ⊢ Eq (l.prod x) (List.map (⇑(LocallyConstant.evalMonoidHom x)) l).prod
  -/
  rw [← map_list_prod (LocallyConstant.evalMonoidHom x) l]
  /-
    I : Type u_1
    C : Set (I → Bool)
    x : ↑C
    l : List (LocallyConstant (↑C) Int)
    ⊢ Eq (l.prod x) ((LocallyConstant.evalMonoidHom x) l.prod)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem factors_prod_eq_basis_of_eq {x y : (π C fun x ↦ x ∈ s)} (h : y = x) :
    (factors C s x).prod y = 1 := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Eq y x
    ⊢ Eq ((Profinite.NobelingProof.factors C s x).prod y) 1
  -/
  rw [list_prod_apply (π C (· ∈ s)) y _]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Eq y x
    ⊢ Eq (List.map (⇑(LocallyConstant.evalMonoidHom y)) (Profinite.NobelingProof.f …
  -/
  apply List.prod_eq_one
  /-
    case hl
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Eq y x
    ⊢ ∀ (x_1 : Int), Membership.mem (List.map (⇑(LocallyConstant.evalMonoidHom y)) …
  -/
  simp only [h, List.mem_map, LocallyConstant.evalMonoidHom, factors]
  /-
    case hl
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Eq y x
    ⊢ ∀ (x_1 : Int), (Exists fun a => And (Exists fun a_1 => And (Membership.mem ( …
  -/
  rintro _ ⟨a, ⟨b, _, rfl⟩, rfl⟩
  /-
    case hl.intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Eq y x
    b : I
    left✝ : Membership.mem (Finset.sort (fun x1 x2 => GE.ge x1 x2) s) b
    ⊢ Eq (((Pi.evalMonoidHom (fun a => Int) x).comp LocallyConstant.coeFnMonoidHom …
  -/
  dsimp
  /-
    case hl.intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Eq y x
    b : I
    left✝ : Membership.mem (Finset.sort (fun x1 x2 => GE.ge x1 x2) s) b
    ⊢ Eq ((ite (Eq (↑x b) Bool.true) (Profinite.NobelingProof.e (Profinite.Nobelin …
  -/
  split_ifs with hh
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Eq y x
      b : I
      left✝ : Membership.mem (Finset.sort (fun x1 x2 => GE.ge x1 x2) s) b
      hh : Eq (↑x b) Bool.true
      ⊢ Eq ((Profinite.NobelingProof.e (Profinite.NobelingProof.π C fun x => Members …
    -/
  · rw [e, LocallyConstant.coe_mk, if_pos hh]
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Eq y x
      b : I
      left✝ : Membership.mem (Finset.sort (fun x1 x2 => GE.ge x1 x2) s) b
      hh : Not (Eq (↑x b) Bool.true)
      ⊢ Eq ((HSub.hSub 1 (Profinite.NobelingProof.e (Profinite.NobelingProof.π C fun …
    -/
  · rw [LocallyConstant.sub_apply, e, LocallyConstant.coe_mk, LocallyConstant.coe_mk, if_neg hh]
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Eq y x
      b : I
      left✝ : Membership.mem (Finset.sort (fun x1 x2 => GE.ge x1 x2) s) b
      hh : Not (Eq (↑x b) Bool.true)
      ⊢ Eq (HSub.hSub (LocallyConstant.toFun 1 x) 0) 1
    -/
    simp only [LocallyConstant.toFun_eq_coe, LocallyConstant.coe_one, Pi.one_apply, sub_zero]
    /-
      🎉 no goals
    -/


theorem e_mem_of_eq_true {x : (π C (· ∈ s))} {a : I} (hx : x.val a = true) :
    e (π C (· ∈ s)) a ∈ factors C s x := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    a : I
    hx : Eq (↑x a) Bool.true
    ⊢ Membership.mem (Profinite.NobelingProof.factors C s x) (Profinite.NobelingPr …
  -/
  rcases x with ⟨_, z, hz, rfl⟩
  /-
    case mk.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    z : I → Bool
    hz : Membership.mem C z
    hx : Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) z, ⋯⟩ a) …
    ⊢ Membership.mem (Profinite.NobelingProof.factors C s ⟨Profinite.NobelingProof …
  -/
  simp only [factors, List.mem_map, Finset.mem_sort]
  /-
    case mk.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    z : I → Bool
    hz : Membership.mem C z
    hx : Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) z, ⋯⟩ a) …
    ⊢ Exists fun a_1 => And (Membership.mem s a_1) (Eq (ite (Eq (Profinite.Nobelin …
  -/
  refine ⟨a, ?_, if_pos hx⟩
  /-
    case mk.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    z : I → Bool
    hz : Membership.mem C z
    hx : Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) z, ⋯⟩ a) …
    ⊢ Membership.mem s a
  -/
  aesop (add simp Proj)
  /-
    🎉 no goals
  -/


theorem one_sub_e_mem_of_false {x y : (π C (· ∈ s))} {a : I} (ha : y.val a = true)
    (hx : x.val a = false) : 1 - e (π C (· ∈ s)) a ∈ factors C s x := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    a : I
    ha : Eq (↑y a) Bool.true
    hx : Eq (↑x a) Bool.false
    ⊢ Membership.mem (Profinite.NobelingProof.factors C s x) (HSub.hSub 1 (Profini …
  -/
  simp only [factors, List.mem_map, Finset.mem_sort]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    a : I
    ha : Eq (↑y a) Bool.true
    hx : Eq (↑x a) Bool.false
    ⊢ Exists fun a_1 => And (Membership.mem s a_1) (Eq (ite (Eq (↑x a_1) Bool.true …
  -/
  use a
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    a : I
    ha : Eq (↑y a) Bool.true
    hx : Eq (↑x a) Bool.false
    ⊢ And (Membership.mem s a) (Eq (ite (Eq (↑x a) Bool.true) (Profinite.NobelingP …
  -/
  simp only [hx, ite_false, and_true]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    a : I
    ha : Eq (↑y a) Bool.true
    hx : Eq (↑x a) Bool.false
    ⊢ And (Membership.mem s a) (Eq (ite (Eq Bool.false Bool.true) (Profinite.Nobel …
  -/
  rcases y with ⟨_, z, hz, rfl⟩
  /-
    case h.mk.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    a : I
    hx : Eq (↑x a) Bool.false
    z : I → Bool
    hz : Membership.mem C z
    ha : Eq (↑⟨Profinite.NobelingProof.Proj (fun x => Membership.mem s x) z, ⋯⟩ a) …
    ⊢ And (Membership.mem s a) (Eq (ite (Eq Bool.false Bool.true) (Profinite.Nobel …
  -/
  aesop (add simp Proj)
  /-
    🎉 no goals
  -/


theorem factors_prod_eq_basis_of_ne {x y : (π C (· ∈ s))} (h : y ≠ x) :
    (factors C s x).prod y = 0 := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Ne y x
    ⊢ Eq ((Profinite.NobelingProof.factors C s x).prod y) 0
  -/
  rw [list_prod_apply (π C (· ∈ s)) y _]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Ne y x
    ⊢ Eq (List.map (⇑(LocallyConstant.evalMonoidHom y)) (Profinite.NobelingProof.f …
  -/
  apply List.prod_eq_zero
  /-
    case a
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Ne y x
    ⊢ Membership.mem (List.map (⇑(LocallyConstant.evalMonoidHom y)) (Profinite.Nob …
  -/
  simp only [List.mem_map]
  /-
    case a
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Ne y x
    ⊢ Exists fun a => And (Membership.mem (Profinite.NobelingProof.factors C s x)  …
  -/
  obtain ⟨a, ha⟩ : ∃ a, y.val a ≠ x.val a := by contrapose! h; ext; apply h
  /-
    case a.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    h : Ne y x
    a : I
    ha : Ne (↑y a) (↑x a)
    ⊢ Exists fun a => And (Membership.mem (Profinite.NobelingProof.factors C s x)  …
  -/
  cases hx : x.val a
    /-
      case a.intro.false
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Ne y x
      a : I
      ha : Ne (↑y a) (↑x a)
      hx : Eq (↑x a) Bool.false
      ⊢ Exists fun a => And (Membership.mem (Profinite.NobelingProof.factors C s x)  …
    -/
  · rw [hx, ne_eq, Bool.not_eq_false] at ha
    /-
      case a.intro.false
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Ne y x
      a : I
      ha : Eq (↑y a) Bool.true
      hx : Eq (↑x a) Bool.false
      ⊢ Exists fun a => And (Membership.mem (Profinite.NobelingProof.factors C s x)  …
    -/
    refine ⟨1 - (e (π C (· ∈ s)) a), ⟨one_sub_e_mem_of_false _ _ ha hx, ?_⟩⟩
    rw [e, LocallyConstant.evalMonoidHom_apply, LocallyConstant.sub_apply,
      LocallyConstant.coe_one, Pi.one_apply, LocallyConstant.coe_mk, if_pos ha, sub_self]
    /-
      case a.intro.true
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Ne y x
      a : I
      ha : Ne (↑y a) (↑x a)
      hx : Eq (↑x a) Bool.true
      ⊢ Exists fun a => And (Membership.mem (Profinite.NobelingProof.factors C s x)  …
    -/
  · refine ⟨e (π C (· ∈ s)) a, ⟨e_mem_of_eq_true _ _ hx, ?_⟩⟩
    /-
      case a.intro.true
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Ne y x
      a : I
      ha : Ne (↑y a) (↑x a)
      hx : Eq (↑x a) Bool.true
      ⊢ Eq ((LocallyConstant.evalMonoidHom y) (Profinite.NobelingProof.e (Profinite. …
    -/
    rw [hx] at ha
    /-
      case a.intro.true
      I : Type u
      C : Set (I → Bool)
      inst✝ : LinearOrder I
      s : Finset I
      x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
      h : Ne y x
      a : I
      ha : Ne (↑y a) Bool.true
      hx : Eq (↑x a) Bool.true
      ⊢ Eq ((LocallyConstant.evalMonoidHom y) (Profinite.NobelingProof.e (Profinite. …
    -/
    rw [LocallyConstant.evalMonoidHom_apply, e, LocallyConstant.coe_mk, if_neg ha]
    /-
      🎉 no goals
    -/


/-- If `s` is finite, the product of the elements of the list `factors C s x`
is the delta function at `x`. -/
theorem factors_prod_eq_basis (x : π C (· ∈ s)) :
    (factors C s x).prod = spanFinBasis C s x := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    ⊢ Eq (Profinite.NobelingProof.factors C s x).prod (Profinite.NobelingProof.spa …
  -/
  ext y
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    x y : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    ⊢ Eq ((Profinite.NobelingProof.factors C s x).prod y) ((Profinite.NobelingProo …
  -/
  dsimp [spanFinBasis]
  split_ifs with h <;> [exact factors_prod_eq_basis_of_eq _ _ h;
    exact factors_prod_eq_basis_of_ne _ _ h]


theorem GoodProducts.finsupp_sum_mem_span_eval {a : I} {as : List I}
    (ha : List.Chain' (· > ·) (a :: as)) {c : Products I →₀ ℤ}
    (hc : (c.support : Set (Products I)) ⊆ {m | m.val ≤ as}) :
    (Finsupp.sum c fun a_1 b ↦ e (π C (· ∈ s)) a * b • Products.eval (π C (· ∈ s)) a_1) ∈
      Submodule.span ℤ (Products.eval (π C (· ∈ s)) '' {m | m.val ≤ a :: as}) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  apply Submodule.finsupp_sum_mem
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    ⊢ ∀ (c_1 : Profinite.NobelingProof.Products I), Ne (c c_1) 0 → Membership.mem  …
  -/
  intro m hm
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  have hsm := (LinearMap.mulLeft ℤ (e (π C (· ∈ s)) a)).map_smul
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    hsm : ∀ (c : Int) (x : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => …
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  dsimp at hsm
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    hsm : ∀ (c : Int) (x : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => …
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  rw [hsm]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    hsm : ∀ (c : Int) (x : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => …
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  apply Submodule.smul_mem
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    hsm : ∀ (c : Int) (x : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => …
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  apply Submodule.subset_span
  have hmas : m.val ≤ as := by
    apply hc
    simpa only [Finset.mem_coe, Finsupp.mem_support_iff] using hm
  /-
    case h.h.a
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    hsm : ∀ (c : Int) (x : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => …
    hmas : LE.le (↑m) as
    ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval (Profinite. …
  -/
  refine ⟨⟨a :: m.val, ha.cons_of_le m.prop hmas⟩, ⟨List.cons_le_cons a hmas, ?_⟩⟩
  /-
    case h.h.a
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    s : Finset I
    a : I
    as : List I
    ha : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a as)
    c : Finsupp (Profinite.NobelingProof.Products I) Int
    hc : HasSubset.Subset (↑c.support) (setOf fun m => LE.le (↑m) as)
    m : Profinite.NobelingProof.Products I
    hm : Ne (c m) 0
    hsm : ∀ (c : Int) (x : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => …
    hmas : LE.le (↑m) as
    ⊢ Eq (Profinite.NobelingProof.Products.eval (Profinite.NobelingProof.π C fun x …
  -/
  simp only [Products.eval, List.map, List.prod_cons]
  /-
    🎉 no goals
  -/


/-- If `s` is a finite subset of `I`, then the good products span. -/
theorem GoodProducts.spanFin [WellFoundedLT I] :
    ⊤ ≤ Submodule.span ℤ (Set.range (eval (π C (· ∈ s)))) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    ⊢ LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.GoodPr …
  -/
  rw [span_iff_products]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    ⊢ LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Produc …
  -/
  refine le_trans (spanFinBasis.span C s) ?_
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    ⊢ LE.le (Submodule.span Int (Set.range (Profinite.NobelingProof.spanFinBasis C …
  -/
  rw [Submodule.span_le]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    ⊢ HasSubset.Subset (Set.range (Profinite.NobelingProof.spanFinBasis C s)) ↑(Su …
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    ⊢ Membership.mem (↑(Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
  -/
  rw [← factors_prod_eq_basis]
  /-
    case intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    ⊢ Membership.mem (↑(Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
  -/
  let l := s.sort (·≥·)
  /-
    case intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    s : Finset I
    inst✝ : WellFoundedLT I
    x : ↑(Profinite.NobelingProof.π C fun x => Membership.mem s x)
    l : List I := Finset.sort (fun x1 x2 => GE.ge x1 x2) s
    ⊢ Membership.mem (↑(Submodule.span Int (Set.range (Profinite.NobelingProof.Pro …
  -/
  dsimp [factors]
  suffices l.Chain' (·>·) → (l.map (fun i ↦ if x.val i = true then e (π C (· ∈ s)) i
      else (1 - (e (π C (· ∈ s)) i)))).prod ∈
      Submodule.span ℤ ((Products.eval (π C (· ∈ s))) '' {m | m.val ≤ l}) from
    Submodule.span_mono (Set.image_subset_range _ _) (this (Finset.sort_sorted_gt _).chain')
  induction l with
  | nil =>
    intro _
    apply Submodule.subset_span
    exact ⟨⟨[], List.chain'_nil⟩,⟨Or.inl rfl, rfl⟩⟩
  | cons a as ih =>
    rw [List.map_cons, List.prod_cons]
    intro ha
    specialize ih (by rw [List.chain'_cons'] at ha; exact ha.2)
    rw [Finsupp.mem_span_image_iff_linearCombination] at ih
    simp only [Finsupp.mem_supported, Finsupp.linearCombination_apply] at ih
    obtain ⟨c, hc, hc'⟩ := ih
    rw [← hc']; clear hc'
    have hmap := fun g ↦ map_finsupp_sum (LinearMap.mulLeft ℤ (e (π C (· ∈ s)) a)) c g
    dsimp at hmap ⊢
    split_ifs
    · rw [hmap]
      exact finsupp_sum_mem_span_eval _ _ ha hc
    · ring_nf
      rw [hmap]
      apply Submodule.add_mem
      · apply Submodule.neg_mem
        exact finsupp_sum_mem_span_eval _ _ ha hc
      · apply Submodule.finsupp_sum_mem
        intro m hm
        apply Submodule.smul_mem
        apply Submodule.subset_span
        refine ⟨m, ⟨?_, rfl⟩⟩
        simp only [Set.mem_setOf_eq]
        have hmas : m.val ≤ as :=
          hc (by simpa only [Finset.mem_coe, Finsupp.mem_support_iff] using hm)
        refine le_trans hmas ?_
        cases as with
        | nil => exact (List.nil_lt_cons a []).le
        | cons b bs =>
          apply le_of_lt
          rw [List.chain'_cons] at ha
          have hlex := List.lt.head bs (b :: bs) ha.1
          exact (List.lt_iff_lex_lt _ _).mp hlex


theorem fin_comap_jointlySurjective
    (hC : IsClosed C)
    (f : LocallyConstant C ℤ) : ∃ (s : Finset I)
    (g : LocallyConstant (π C (· ∈ s)) ℤ), f = g.comap ⟨(ProjRestrict C (· ∈ s)),
      continuous_projRestrict _ _⟩ := by
  obtain ⟨J, g, h⟩ := @Profinite.exists_locallyConstant (Finset I)ᵒᵖ _ _ _
    (spanCone hC.isCompact) ℤ
    (spanCone_isLimit hC.isCompact) f
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝ : LinearOrder I
    hC : IsClosed C
    f : LocallyConstant (↑C) Int
    J : Opposite (Finset I)
    g : LocallyConstant (↑((Profinite.NobelingProof.spanFunctor ⋯).obj J).toTop) Int
    h : Eq f (LocallyConstant.comap ((Profinite.NobelingProof.spanCone ⋯).π.app J) …
    ⊢ Exists fun s => Exists fun g => Eq f (LocallyConstant.comap { toFun := Profi …
  -/
  exact ⟨(Opposite.unop J), g, h⟩
  /-
    🎉 no goals
  -/


/-- The good products span all of `LocallyConstant C ℤ` if `C` is closed. -/
theorem GoodProducts.span [WellFoundedLT I] (hC : IsClosed C) :
    ⊤ ≤ Submodule.span ℤ (Set.range (eval C)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    hC : IsClosed C
    ⊢ LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.GoodPr …
  -/
  rw [span_iff_products]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    hC : IsClosed C
    ⊢ LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Produc …
  -/
  intro f _
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    hC : IsClosed C
    f : LocallyConstant (↑C) Int
    a✝ : Membership.mem Top.top f
    ⊢ Membership.mem (Submodule.span Int (Set.range (Profinite.NobelingProof.Produ …
  -/
  obtain ⟨K, f', rfl⟩ : ∃ K f', f = πJ C K f' := fin_comap_jointlySurjective C hC f
  refine Submodule.span_mono ?_ <| Submodule.apply_mem_span_image_of_mem_span (πJ C K) <|
    spanFin C K (Submodule.mem_top : f' ∈ ⊤)
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    hC : IsClosed C
    K : Finset I
    f' : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem K  …
    a✝ : Membership.mem Top.top ((Profinite.NobelingProof.πJ C K) f')
    ⊢ HasSubset.Subset (Set.image (⇑(Profinite.NobelingProof.πJ C K)) (Set.range ( …
  -/
  rintro l ⟨y, ⟨m, rfl⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    hC : IsClosed C
    K : Finset I
    f' : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => Membership.mem K  …
    a✝ : Membership.mem Top.top ((Profinite.NobelingProof.πJ C K) f')
    m : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
    ⊢ Membership.mem (Set.range (Profinite.NobelingProof.Products.eval C)) ((Profi …
  -/
  exact ⟨m.val, eval_eq_πJ C K m.val m.prop⟩
  /-
    🎉 no goals
  -/


/-- A term of `I` regarded as an ordinal. -/
def ord (i : I) : Ordinal := Ordinal.typein ((·<·) : I → I → Prop) i


/-- An ordinal regarded as a term of `I`. -/
noncomputable
def term {o : Ordinal} (ho : o < Ordinal.type ((·<·) : I → I → Prop)) : I :=
  Ordinal.enum ((·<·) : I → I → Prop) ⟨o, ho⟩


theorem term_ord_aux {i : I} (ho : ord I i < Ordinal.type ((·<·) : I → I → Prop)) :
    term I ho = i := by
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    i : I
    ho : LT.lt (Profinite.NobelingProof.ord I i) (Ordinal.type fun x1 x2 => LT.lt  …
    ⊢ Eq (Profinite.NobelingProof.term I ho) i
  -/
  simp only [term, ord, Ordinal.enum_typein]
  /-
    🎉 no goals
  -/


@[simp]
theorem ord_term_aux {o : Ordinal} (ho : o < Ordinal.type ((·<·) : I → I → Prop)) :
    ord I (term I ho) = o := by
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Profinite.NobelingProof.ord I (Profinite.NobelingProof.term I ho)) o
  -/
  simp only [ord, term, Ordinal.typein_enum]
  /-
    🎉 no goals
  -/


theorem ord_term {o : Ordinal} (ho : o < Ordinal.type ((·<·) : I → I → Prop)) (i : I) :
    ord I i = o ↔ term I ho = i := by
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    i : I
    ⊢ Iff (Eq (Profinite.NobelingProof.ord I i) o) (Eq (Profinite.NobelingProof.te …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      i : I
      h : Eq (Profinite.NobelingProof.ord I i) o
      ⊢ Eq (Profinite.NobelingProof.term I ho) i
    -/
  · subst h
    /-
      case refine_1
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      i : I
      ho : LT.lt (Profinite.NobelingProof.ord I i) (Ordinal.type fun x1 x2 => LT.lt  …
      ⊢ Eq (Profinite.NobelingProof.term I ho) i
    -/
    exact term_ord_aux ho
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      i : I
      h : Eq (Profinite.NobelingProof.term I ho) i
      ⊢ Eq (Profinite.NobelingProof.ord I i) o
    -/
  · subst h
    /-
      case refine_2
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      ⊢ Eq (Profinite.NobelingProof.ord I (Profinite.NobelingProof.term I ho)) o
    -/
    exact ord_term_aux ho
    /-
      🎉 no goals
    -/


/-- A predicate saying that `C` is "small" enough to satisfy the inductive hypothesis. -/
def contained (o : Ordinal) : Prop := ∀ f, f ∈ C → ∀ (i : I), f i = true → ord I i < o


variable (I) in
/--
The predicate on ordinals which we prove by induction, see `GoodProducts.P0`,
`GoodProducts.Plimit` and `GoodProducts.linearIndependentAux` in the section `Induction` below
-/
def P (o : Ordinal) : Prop :=
  o ≤ Ordinal.type (·<· : I → I → Prop) →
  (∀ (C : Set (I → Bool)), IsClosed C → contained C o →
    LinearIndependent ℤ (GoodProducts.eval C))


theorem Products.prop_of_isGood_of_contained  {l : Products I} (o : Ordinal) (h : l.isGood C)
    (hsC : contained C o) (i : I) (hi : i ∈ l.val) : ord I i < o := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    ⊢ LT.lt (Profinite.NobelingProof.ord I i) o
  -/
  by_contra h'
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (LT.lt (Profinite.NobelingProof.ord I i) o)
    ⊢ False
  -/
  apply h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (LT.lt (Profinite.NobelingProof.ord I i) o)
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  suffices eval C l = 0 by simp [this, Submodule.zero_mem]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (LT.lt (Profinite.NobelingProof.ord I i) o)
    ⊢ Eq (Profinite.NobelingProof.Products.eval C l) 0
  -/
  ext x
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (LT.lt (Profinite.NobelingProof.ord I i) o)
    x : ↑C
    ⊢ Eq ((Profinite.NobelingProof.Products.eval C l) x) (0 x)
  -/
  simp only [eval_eq, LocallyConstant.coe_zero, Pi.zero_apply, ite_eq_right_iff, one_ne_zero]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    h' : Not (LT.lt (Profinite.NobelingProof.ord I i) o)
    x : ↑C
    ⊢ (∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true) → False
  -/
  contrapose! h'
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.Products.isGood C l
    hsC : Profinite.NobelingProof.contained C o
    i : I
    hi : Membership.mem (↑l) i
    x : ↑C
    h' : And (∀ (i : I), Membership.mem (↑l) i → Eq (↑x i) Bool.true) (Not False)
    ⊢ LT.lt (Profinite.NobelingProof.ord I i) o
  -/
  exact hsC x.val x.prop i (h'.1 i hi)
  /-
    🎉 no goals
  -/


instance : Subsingleton (LocallyConstant (∅ : Set (I → Bool)) ℤ) :=
  subsingleton_iff.mpr (fun _ _ ↦ LocallyConstant.ext isEmptyElim)


instance : IsEmpty { l // Products.isGood (∅ : Set (I → Bool)) l } :=
  isEmpty_iff.mpr fun ⟨l, hl⟩ ↦ hl <| by
    /-
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      x✝ : Subtype fun l => Profinite.NobelingProof.Products.isGood EmptyCollection. …
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood EmptyCollection.emptyCollection l
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    rw [subsingleton_iff.mp inferInstance (Products.eval ∅ l) 0]
    /-
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      x✝ : Subtype fun l => Profinite.NobelingProof.Products.isGood EmptyCollection. …
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood EmptyCollection.emptyCollection l
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    exact Submodule.zero_mem _
    /-
      🎉 no goals
    -/


theorem GoodProducts.linearIndependentEmpty {I} [LinearOrder I] :
    LinearIndependent ℤ (eval (∅ : Set (I → Bool))) := linearIndependent_empty_type


/-- The empty list as a `Products` -/
                                         /-
                                           I : Type u
                                           C : Set (I → Bool)
                                           inst✝¹ : LinearOrder I
                                           inst✝ : WellFoundedLT I
                                           ⊢ List.Chain' (fun x1 x2 => GT.gt x1 x2) List.nil
                                         -/
def Products.nil : Products I := ⟨[], by simp only [List.chain'_nil]⟩
                                         /-
                                           🎉 no goals
                                         -/


theorem Products.lt_nil_empty {I} [LinearOrder I] : { m : Products I | m < Products.nil } = ∅ := by
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    ⊢ Eq (setOf fun m => LT.lt m Profinite.NobelingProof.Products.nil) EmptyCollec …
  -/
  ext ⟨m, hm⟩
  /-
    case h.mk
    I : Type u_1
    inst✝ : LinearOrder I
    m : List I
    hm : List.Chain' (fun x1 x2 => GT.gt x1 x2) m
    ⊢ Iff (Membership.mem (setOf fun m => LT.lt m Profinite.NobelingProof.Products …
  -/
  refine ⟨fun h ↦ ?_, by tauto⟩
  /-
    case h.mk
    I : Type u_1
    inst✝ : LinearOrder I
    m : List I
    hm : List.Chain' (fun x1 x2 => GT.gt x1 x2) m
    h : Membership.mem (setOf fun m => LT.lt m Profinite.NobelingProof.Products.ni …
    ⊢ Membership.mem EmptyCollection.emptyCollection ⟨m, hm⟩
  -/
  simp only [Set.mem_setOf_eq, lt_iff_lex_lt, nil, List.Lex.not_nil_right] at h
  /-
    🎉 no goals
  -/


instance {α : Type*} [TopologicalSpace α] [Nonempty α] : Nontrivial (LocallyConstant α ℤ) :=
  ⟨0, 1, ne_of_apply_ne DFunLike.coe <| (Function.const_injective (β := ℤ)).ne zero_ne_one⟩


theorem Products.isGood_nil {I} [LinearOrder I] :
    Products.isGood ({fun _ ↦ false} : Set (I → Bool)) Products.nil := by
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    ⊢ Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bool.f …
  -/
  intro h
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    h : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pro …
    ⊢ False
  -/
  simp [Products.eval, Products.nil] at h
  /-
    🎉 no goals
  -/


theorem Products.span_nil_eq_top {I} [LinearOrder I] :
    Submodule.span ℤ (eval ({fun _ ↦ false} : Set (I → Bool)) '' {nil}) = ⊤ := by
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    ⊢ Eq (Submodule.span Int (Set.image (Profinite.NobelingProof.Products.eval (Si …
  -/
  rw [Set.image_singleton, eq_top_iff]
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    ⊢ LE.le Top.top (Submodule.span Int (Singleton.singleton (Profinite.NobelingPr …
  -/
  intro f _
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    f : LocallyConstant (↑(Singleton.singleton fun x => Bool.false)) Int
    a✝ : Membership.mem Top.top f
    ⊢ Membership.mem (Submodule.span Int (Singleton.singleton (Profinite.NobelingP …
  -/
  rw [Submodule.mem_span_singleton]
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    f : LocallyConstant (↑(Singleton.singleton fun x => Bool.false)) Int
    a✝ : Membership.mem Top.top f
    ⊢ Exists fun a => Eq (HSMul.hSMul a (Profinite.NobelingProof.Products.eval (Si …
  -/
  refine ⟨f default, ?_⟩
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    f : LocallyConstant (↑(Singleton.singleton fun x => Bool.false)) Int
    a✝ : Membership.mem Top.top f
    ⊢ Eq (HSMul.hSMul (f Inhabited.default) (Profinite.NobelingProof.Products.eval …
  -/
  simp only [eval, List.map, List.prod_nil, zsmul_eq_mul, mul_one, Products.nil]
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    f : LocallyConstant (↑(Singleton.singleton fun x => Bool.false)) Int
    a✝ : Membership.mem Top.top f
    ⊢ Eq (↑(f Inhabited.default)) f
  -/
  ext x
  /-
    case h
    I : Type u_1
    inst✝ : LinearOrder I
    f : LocallyConstant (↑(Singleton.singleton fun x => Bool.false)) Int
    a✝ : Membership.mem Top.top f
    x : ↑(Singleton.singleton fun x => Bool.false)
    ⊢ Eq (↑(f Inhabited.default) x) (f x)
  -/
  obtain rfl : x = default := by simp only [Set.default_coe_singleton, eq_iff_true_of_subsingleton]
  /-
    case h
    I : Type u_1
    inst✝ : LinearOrder I
    f : LocallyConstant (↑(Singleton.singleton fun x => Bool.false)) Int
    a✝ : Membership.mem Top.top f
    ⊢ Eq (↑(f Inhabited.default) Inhabited.default) (f Inhabited.default)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- There is a unique `GoodProducts` for the singleton `{fun _ ↦ false}`. -/
noncomputable
instance : Unique { l // Products.isGood ({fun _ ↦ false} : Set (I → Bool)) l } where
  default := ⟨Products.nil, Products.isGood_nil⟩
  uniq := by
    /-
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      ⊢ ∀ (a : Subtype fun l => Profinite.NobelingProof.Products.isGood (Singleton.s …
    -/
    intro ⟨⟨l, hl⟩, hll⟩
    /-
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      ⊢ Eq ⟨⟨l, hl⟩, hll⟩ Inhabited.default
    -/
    ext
    /-
      case a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      ⊢ Eq ↑⟨⟨l, hl⟩, hll⟩ ↑Inhabited.default
    -/
    apply Subtype.ext
    /-
      case a.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      ⊢ Eq ↑↑⟨⟨l, hl⟩, hll⟩ ↑↑Inhabited.default
    -/
    apply (List.Lex.nil_left_or_eq_nil l (r := (·<·))).resolve_left
    /-
      case a.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      ⊢ Not (List.Lex (fun x1 x2 => LT.lt x1 x2) List.nil l)
    -/
    intro _
    /-
      case a.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      a✝ : List.Lex (fun x1 x2 => LT.lt x1 x2) List.nil l
      ⊢ False
    -/
    apply hll
    have he : {Products.nil} ⊆ {m | m < ⟨l,hl⟩} := by
      simpa only [Products.nil, Products.lt_iff_lex_lt, Set.singleton_subset_iff, Set.mem_setOf_eq]
    /-
      case a.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      a✝ : List.Lex (fun x1 x2 => LT.lt x1 x2) List.nil l
      he : HasSubset.Subset (Singleton.singleton Profinite.NobelingProof.Products.ni …
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    apply Submodule.span_mono (Set.image_subset _ he)
    /-
      case a.a.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      a✝ : List.Lex (fun x1 x2 => LT.lt x1 x2) List.nil l
      he : HasSubset.Subset (Singleton.singleton Profinite.NobelingProof.Products.ni …
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    rw [Products.span_nil_eq_top]
    /-
      case a.a.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      l : List I
      hl : List.Chain' (fun x1 x2 => GT.gt x1 x2) l
      hll : Profinite.NobelingProof.Products.isGood (Singleton.singleton fun x => Bo …
      a✝ : List.Lex (fun x1 x2 => LT.lt x1 x2) List.nil l
      he : HasSubset.Subset (Singleton.singleton Profinite.NobelingProof.Products.ni …
      ⊢ Membership.mem Top.top (Profinite.NobelingProof.Products.eval (Singleton.sin …
    -/
    exact Submodule.mem_top
    /-
      🎉 no goals
    -/


instance (α : Type*) [TopologicalSpace α] : NoZeroSMulDivisors ℤ (LocallyConstant α ℤ) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ NoZeroSMulDivisors Int (LocallyConstant α Int)
  -/
  constructor
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ ∀ {c : Int} {x : LocallyConstant α Int}, Eq (HSMul.hSMul c x) 0 → Or (Eq c 0 …
  -/
  intro c f h
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    c : Int
    f : LocallyConstant α Int
    h : Eq (HSMul.hSMul c f) 0
    ⊢ Or (Eq c 0) (Eq f 0)
  -/
  rw [or_iff_not_imp_left]
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    c : Int
    f : LocallyConstant α Int
    h : Eq (HSMul.hSMul c f) 0
    ⊢ Not (Eq c 0) → Eq f 0
  -/
  intro hc
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    c : Int
    f : LocallyConstant α Int
    h : Eq (HSMul.hSMul c f) 0
    hc : Not (Eq c 0)
    ⊢ Eq f 0
  -/
  ext x
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero.h
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    c : Int
    f : LocallyConstant α Int
    h : Eq (HSMul.hSMul c f) 0
    hc : Not (Eq c 0)
    x : α
    ⊢ Eq (f x) (0 x)
  -/
  apply mul_right_injective₀ hc
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero.h.a
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    c : Int
    f : LocallyConstant α Int
    h : Eq (HSMul.hSMul c f) 0
    hc : Not (Eq c 0)
    x : α
    ⊢ Eq ((fun x => HMul.hMul c x) (f x)) ((fun x => HMul.hMul c x) (0 x))
  -/
  simp [LocallyConstant.ext_iff] at h
  /-
    case eq_zero_or_eq_zero_of_smul_eq_zero.h.a
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    α : Type u_1
    inst✝ : TopologicalSpace α
    c : Int
    f : LocallyConstant α Int
    hc : Not (Eq c 0)
    x : α
    h : ∀ (x : α), Or (Eq (↑c x) 0) (Eq (f x) 0)
    ⊢ Eq ((fun x => HMul.hMul c x) (f x)) ((fun x => HMul.hMul c x) (0 x))
  -/
  simpa [LocallyConstant.ext_iff] using h x
  /-
    🎉 no goals
  -/


theorem GoodProducts.linearIndependentSingleton {I} [LinearOrder I] :
    LinearIndependent ℤ (eval ({fun _ ↦ false} : Set (I → Bool))) := by
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Singleton. …
  -/
  refine linearIndependent_unique (eval ({fun _ ↦ false} : Set (I → Bool))) ?_
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    ⊢ Ne (Profinite.NobelingProof.GoodProducts.eval (Singleton.singleton fun x =>  …
  -/
  simp [eval, Products.eval, Products.nil, default]
  /-
    🎉 no goals
  -/


theorem contained_eq_proj (o : Ordinal) (h : contained C o) :
    C = π C (ord I · < o) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.contained C o
    ⊢ Eq C (Profinite.NobelingProof.π C fun x => LT.lt (Profinite.NobelingProof.or …
  -/
  have := proj_prop_eq_self C (ord I · < o)
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.contained C o
    this : (∀ (i : I) (x : I → Bool), Membership.mem C x → Ne (x i) Bool.false → L …
    ⊢ Eq C (Profinite.NobelingProof.π C fun x => LT.lt (Profinite.NobelingProof.or …
  -/
  simp only [ne_eq, Bool.not_eq_false, π] at this
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    h : Profinite.NobelingProof.contained C o
    this : (∀ (i : I) (x : I → Bool), Membership.mem C x → Eq (x i) Bool.true → LT …
    ⊢ Eq C (Profinite.NobelingProof.π C fun x => LT.lt (Profinite.NobelingProof.or …
  -/
  exact (this (fun i x hx ↦ h x hx i)).symm
  /-
    🎉 no goals
  -/


theorem isClosed_proj (o : Ordinal) (hC : IsClosed C) : IsClosed (π C (ord I · < o)) :=
  (continuous_proj (ord I · < o)).isClosedMap C hC


theorem contained_proj (o : Ordinal) : contained (π C (ord I · < o)) o := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Profinite.NobelingProof.contained (Profinite.NobelingProof.π C fun x => LT.l …
  -/
  intro x ⟨_, _, h⟩ j hj
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    x w✝ : I → Bool
    left✝ : Membership.mem C w✝
    h : Eq (Profinite.NobelingProof.Proj (fun x => LT.lt (Profinite.NobelingProof. …
    j : I
    hj : Eq (x j) Bool.true
    ⊢ LT.lt (Profinite.NobelingProof.ord I j) o
  -/
  aesop (add simp Proj)
  /-
    🎉 no goals
  -/


/-- The `ℤ`-linear map induced by precomposition of the projection `C → π C (ord I · < o)`. -/
@[simps!]
noncomputable
def πs (o : Ordinal) : LocallyConstant (π C (ord I · < o)) ℤ →ₗ[ℤ] LocallyConstant C ℤ :=
  LocallyConstant.comapₗ ℤ ⟨(ProjRestrict C (ord I · < o)), (continuous_projRestrict _ _)⟩


theorem coe_πs (o : Ordinal) (f : LocallyConstant (π C (ord I · < o)) ℤ) :
    πs C o f = f ∘ ProjRestrict C (ord I · < o) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    ⊢ Eq (⇑((Profinite.NobelingProof.πs C o) f)) (Function.comp (⇑f) (Profinite.No …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem injective_πs (o : Ordinal) : Function.Injective (πs C o) :=
  LocallyConstant.comap_injective ⟨_, (continuous_projRestrict _ _)⟩
    (Set.surjective_mapsTo_image_restrict _ _)


/-- The `ℤ`-linear map induced by precomposition of the projection
    `π C (ord I · < o₂) → π C (ord I · < o₁)` for `o₁ ≤ o₂`. -/
@[simps!]
noncomputable
def πs' {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂) :
    LocallyConstant (π C (ord I · < o₁)) ℤ →ₗ[ℤ] LocallyConstant (π C (ord I · < o₂)) ℤ :=
  LocallyConstant.comapₗ ℤ ⟨(ProjRestricts C (fun _ hh ↦ lt_of_lt_of_le hh h)),
    (continuous_projRestricts _ _)⟩


theorem coe_πs' {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂) (f : LocallyConstant (π C (ord I · < o₁)) ℤ) :
    (πs' C h f).toFun = f.toFun ∘ (ProjRestricts C (fun _ hh ↦ lt_of_lt_of_le hh h)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    ⊢ Eq ((Profinite.NobelingProof.πs' C h) f).toFun (Function.comp f.toFun (Profi …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem injective_πs' {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂) : Function.Injective (πs' C h) :=
  LocallyConstant.comap_injective ⟨_, (continuous_projRestricts _ _)⟩
    (surjective_projRestricts _ fun _ hi ↦ lt_of_lt_of_le hi h)


theorem lt_ord_of_lt {l m : Products I} {o : Ordinal} (h₁ : m < l)
    (h₂ : ∀ i ∈ l.val, ord I i < o) : ∀ i ∈ m.val, ord I i < o :=
  List.Sorted.lt_ord_of_lt (List.chain'_iff_pairwise.mp l.2) (List.chain'_iff_pairwise.mp m.2) h₁ h₂


theorem eval_πs {l : Products I} {o : Ordinal} (hlt : ∀ i ∈ l.val, ord I i < o) :
    πs C o (l.eval (π C (ord I · < o))) = l.eval C := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    hlt : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I  …
    ⊢ Eq ((Profinite.NobelingProof.πs C o) (Profinite.NobelingProof.Products.eval  …
  -/
  simpa only [← LocallyConstant.coe_inj] using evalFacProp C (ord I · < o) hlt
  /-
    🎉 no goals
  -/


theorem eval_πs' {l : Products I} {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂)
    (hlt : ∀ i ∈ l.val, ord I i < o₁) :
    πs' C h (l.eval (π C (ord I · < o₁))) = l.eval (π C (ord I · < o₂)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hlt : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I  …
    ⊢ Eq ((Profinite.NobelingProof.πs' C h) (Profinite.NobelingProof.Products.eval …
  -/
  rw [← LocallyConstant.coe_inj, ← LocallyConstant.toFun_eq_coe]
  exact evalFacProps C (fun (i : I) ↦ ord I i < o₁) (fun (i : I) ↦ ord I i < o₂) hlt
    (fun _ hh ↦ lt_of_lt_of_le hh h)


theorem eval_πs_image {l : Products I} {o : Ordinal}
    (hl : ∀ i ∈ l.val, ord I i < o) : eval C '' { m | m < l } =
    (πs C o) '' (eval (π C (ord I · < o)) '' { m | m < l }) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    ⊢ Eq (Set.image (Profinite.NobelingProof.Products.eval C) (setOf fun m => LT.l …
  -/
  ext f
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑C) Int
    ⊢ Iff (Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (se …
  -/
  simp only [Set.mem_image, Set.mem_setOf_eq, exists_exists_and_eq_and]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑C) Int
    ⊢ Iff (Exists fun x => And (LT.lt x l) (Eq (Profinite.NobelingProof.Products.e …
  -/
  apply exists_congr; intro m
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑C) Int
    m : Profinite.NobelingProof.Products I
    ⊢ Iff (And (LT.lt m l) (Eq (Profinite.NobelingProof.Products.eval C m) f)) (An …
  -/
  apply and_congr_right; intro hm
  /-
    case h.h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o : Ordinal.{u}
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑C) Int
    m : Profinite.NobelingProof.Products I
    hm : LT.lt m l
    ⊢ Iff (Eq (Profinite.NobelingProof.Products.eval C m) f) (Eq ((Profinite.Nobel …
  -/
  rw [eval_πs C (lt_ord_of_lt hm hl)]
  /-
    🎉 no goals
  -/


theorem eval_πs_image' {l : Products I} {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂)
    (hl : ∀ i ∈ l.val, ord I i < o₁) : eval (π C (ord I · < o₂)) '' { m | m < l } =
    (πs' C h) '' (eval (π C (ord I · < o₁)) '' { m | m < l }) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    ⊢ Eq (Set.image (Profinite.NobelingProof.Products.eval (Profinite.NobelingProo …
  -/
  ext f
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    ⊢ Iff (Membership.mem (Set.image (Profinite.NobelingProof.Products.eval (Profi …
  -/
  simp only [Set.mem_image, Set.mem_setOf_eq, exists_exists_and_eq_and]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    ⊢ Iff (Exists fun x => And (LT.lt x l) (Eq (Profinite.NobelingProof.Products.e …
  -/
  apply exists_congr; intro m
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    m : Profinite.NobelingProof.Products I
    ⊢ Iff (And (LT.lt m l) (Eq (Profinite.NobelingProof.Products.eval (Profinite.N …
  -/
  apply and_congr_right; intro hm
  /-
    case h.h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
    f : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    m : Profinite.NobelingProof.Products I
    hm : LT.lt m l
    ⊢ Iff (Eq (Profinite.NobelingProof.Products.eval (Profinite.NobelingProof.π C  …
  -/
  rw [eval_πs' C h (lt_ord_of_lt hm hl)]
  /-
    🎉 no goals
  -/


theorem head_lt_ord_of_isGood [Inhabited I] {l : Products I} {o : Ordinal}
    (h : l.isGood (π C (ord I · < o))) (hn : l.val ≠ []) : ord I (l.val.head!) < o :=
  prop_of_isGood C (ord I · < o) h l.val.head! (List.head!_mem_self hn)


/--
If `l` is good w.r.t. `π C (ord I · < o₁)` and `o₁ ≤ o₂`, then it is good w.r.t.
`π C (ord I · < o₂)`
-/
theorem isGood_mono {l : Products I} {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂)
    (hl : l.isGood (π C (ord I · < o₁))) : l.isGood (π C (ord I · < o₂)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
    ⊢ Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun x = …
  -/
  intro hl'
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    l : Profinite.NobelingProof.Products I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
    hl' : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.P …
    ⊢ False
  -/
  apply hl
  rwa [eval_πs_image' C h (prop_of_isGood  C _ hl), ← eval_πs' C h (prop_of_isGood  C _ hl),
    Submodule.apply_mem_span_image_iff_mem_span (injective_πs' C h)] at hl'


/--
The image of the `GoodProducts` for `π C (ord I · < o)` in `LocallyConstant C ℤ`. The name `smaller`
refers to the setting in which we will use this, when we are mapping in `GoodProducts` from a
smaller set, i.e. when `o` is a smaller ordinal than the one `C` is "contained" in.
-/
def smaller (o : Ordinal) : Set (LocallyConstant C ℤ) :=
  (πs C o) '' (range (π C (ord I · < o)))


/--
The map from the image of the `GoodProducts` in `LocallyConstant (π C (ord I · < o)) ℤ` to
`smaller C o`
-/
noncomputable
def range_equiv_smaller_toFun (o : Ordinal) (x : range (π C (ord I · < o))) : smaller C o :=
  ⟨πs C o ↑x, x.val, x.property, rfl⟩


theorem range_equiv_smaller_toFun_bijective (o : Ordinal) :
    Function.Bijective (range_equiv_smaller_toFun C o) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Function.Bijective (Profinite.NobelingProof.GoodProducts.range_equiv_smaller …
  -/
  dsimp (config := { unfoldPartialApp := true }) [range_equiv_smaller_toFun]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Function.Bijective fun x => ⟨(Profinite.NobelingProof.πs C o) ↑x, ⋯⟩
  -/
  refine ⟨fun a b hab ↦ ?_, fun ⟨a, b, hb⟩ ↦ ?_⟩
    /-
      case refine_1
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      a b : ↑(Profinite.NobelingProof.GoodProducts.range (Profinite.NobelingProof.π  …
      hab : Eq ((fun x => ⟨(Profinite.NobelingProof.πs C o) ↑x, ⋯⟩) a) ((fun x => ⟨( …
      ⊢ Eq a b
    -/
  · ext1
    /-
      case refine_1.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      a b : ↑(Profinite.NobelingProof.GoodProducts.range (Profinite.NobelingProof.π  …
      hab : Eq ((fun x => ⟨(Profinite.NobelingProof.πs C o) ↑x, ⋯⟩) a) ((fun x => ⟨( …
      ⊢ Eq ↑a ↑b
    -/
    simp only [Subtype.mk.injEq] at hab
    /-
      case refine_1.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      a b : ↑(Profinite.NobelingProof.GoodProducts.range (Profinite.NobelingProof.π  …
      hab : Eq ((Profinite.NobelingProof.πs C o) ↑a) ((Profinite.NobelingProof.πs C  …
      ⊢ Eq ↑a ↑b
    -/
    exact injective_πs C o hab
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      x✝ : ↑(Profinite.NobelingProof.GoodProducts.smaller C o)
      a : LocallyConstant (↑C) Int
      b : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      hb : And (Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinit …
      ⊢ Exists fun a_1 => Eq ((fun x => ⟨(Profinite.NobelingProof.πs C o) ↑x, ⋯⟩) a_ …
    -/
  · use ⟨b, hb.1⟩
    /-
      case h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      x✝ : ↑(Profinite.NobelingProof.GoodProducts.smaller C o)
      a : LocallyConstant (↑C) Int
      b : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      hb : And (Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinit …
      ⊢ Eq ((fun x => ⟨(Profinite.NobelingProof.πs C o) ↑x, ⋯⟩) ⟨b, ⋯⟩) ⟨a, ⋯⟩
    -/
    simpa only [Subtype.mk.injEq] using hb.2
    /-
      🎉 no goals
    -/


/--
The equivalence from the image of the `GoodProducts` in `LocallyConstant (π C (ord I · < o)) ℤ` to
`smaller C o`
-/
noncomputable
def range_equiv_smaller (o : Ordinal) : range (π C (ord I · < o)) ≃ smaller C o :=
  Equiv.ofBijective (range_equiv_smaller_toFun C o) (range_equiv_smaller_toFun_bijective C o)


theorem smaller_factorization (o : Ordinal) :
    (fun (p : smaller C o) ↦ p.1) ∘ (range_equiv_smaller C o).toFun =
                                                                 /-
                                                                   I : Type u
                                                                   C : Set (I → Bool)
                                                                   inst✝¹ : LinearOrder I
                                                                   inst✝ : WellFoundedLT I
                                                                   o : Ordinal.{u}
                                                                   ⊢ Eq (Function.comp (fun p => ↑p) (Profinite.NobelingProof.GoodProducts.range_ …
                                                                 -/
    (πs C o) ∘ (fun (p : range (π C (ord I · < o))) ↦ p.1) := by rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem linearIndependent_iff_smaller (o : Ordinal) :
    LinearIndependent ℤ (GoodProducts.eval (π C (ord I · < o))) ↔
    LinearIndependent ℤ (fun (p : smaller C o) ↦ p.1) := by
  rw [GoodProducts.linearIndependent_iff_range,
    ← LinearMap.linearIndependent_iff (πs C o)
    (LinearMap.ker_eq_bot_of_injective (injective_πs _ _)), ← smaller_factorization C o]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Iff (LinearIndependent Int (Function.comp (fun p => ↑p) (Profinite.NobelingP …
  -/
  exact linearIndependent_equiv _
  /-
    🎉 no goals
  -/


theorem smaller_mono {o₁ o₂ : Ordinal} (h : o₁ ≤ o₂) : smaller C o₁ ⊆ smaller C o₂ := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    ⊢ HasSubset.Subset (Profinite.NobelingProof.GoodProducts.smaller C o₁) (Profin …
  -/
  rintro f ⟨g, hg, rfl⟩
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    g : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hg : Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinite.Nob …
    ⊢ Membership.mem (Profinite.NobelingProof.GoodProducts.smaller C o₂) ((Profini …
  -/
  simp only [smaller, Set.mem_image]
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    g : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hg : Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinite.Nob …
    ⊢ Exists fun x => And (Membership.mem (Profinite.NobelingProof.GoodProducts.ra …
  -/
  use πs' C h g
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    g : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hg : Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinite.Nob …
    ⊢ And (Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinite.N …
  -/
  obtain ⟨⟨l, gl⟩, rfl⟩ := hg
  /-
    case h.intro.mk
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o₁ o₂ : Ordinal.{u}
    h : LE.le o₁ o₂
    l : Profinite.NobelingProof.Products I
    gl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
    ⊢ And (Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinite.N …
  -/
  refine ⟨?_, ?_⟩
    /-
      case h.intro.mk.refine_1
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o₁ o₂ : Ordinal.{u}
      h : LE.le o₁ o₂
      l : Profinite.NobelingProof.Products I
      gl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Membership.mem (Profinite.NobelingProof.GoodProducts.range (Profinite.Nobeli …
    -/
  · use ⟨l, Products.isGood_mono C h gl⟩
    /-
      case h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o₁ o₂ : Ordinal.{u}
      h : LE.le o₁ o₂
      l : Profinite.NobelingProof.Products I
      gl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Eq (Profinite.NobelingProof.GoodProducts.eval (Profinite.NobelingProof.π C f …
    -/
    ext x
    /-
      case h.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o₁ o₂ : Ordinal.{u}
      h : LE.le o₁ o₂
      l : Profinite.NobelingProof.Products I
      gl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      x : ↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.NobelingProof.ord  …
      ⊢ Eq ((Profinite.NobelingProof.GoodProducts.eval (Profinite.NobelingProof.π C  …
    -/
    rw [eval, ← Products.eval_πs' _ h (Products.prop_of_isGood  C _ gl), eval]
    /-
      🎉 no goals
    -/
  · rw [← LocallyConstant.coe_inj, coe_πs C o₂, ← LocallyConstant.toFun_eq_coe, coe_πs',
      Function.comp_assoc, projRestricts_comp_projRestrict C _, coe_πs]
    /-
      case h.intro.mk.refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o₁ o₂ : Ordinal.{u}
      h : LE.le o₁ o₂
      l : Profinite.NobelingProof.Products I
      gl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Eq (Function.comp (Profinite.NobelingProof.GoodProducts.eval (Profinite.Nobe …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem Products.limitOrdinal (l : Products I) : l.isGood (π C (ord I · < o)) ↔
    ∃ (o' : Ordinal), o' < o ∧ l.isGood (π C (ord I · < o')) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    l : Profinite.NobelingProof.Products I
    ⊢ Iff (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fu …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨o', ⟨ho', hl⟩⟩ ↦ isGood_mono C (le_of_lt ho') hl⟩
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    l : Profinite.NobelingProof.Products I
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun x …
    ⊢ Exists fun o' => And (LT.lt o' o) (Profinite.NobelingProof.Products.isGood ( …
  -/
  use Finset.sup l.val.toFinset (fun a ↦ Order.succ (ord I a))
  have hslt : Finset.sup l.val.toFinset (fun a ↦ Order.succ (ord I a)) < o := by
    simp only [Finset.sup_lt_iff ho.pos, List.mem_toFinset]
    exact fun b hb ↦ ho.succ_lt (prop_of_isGood C (ord I · < o) h b hb)
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    l : Profinite.NobelingProof.Products I
    h : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun x …
    hslt : LT.lt ((↑l).toFinset.sup fun a => Order.succ (Profinite.NobelingProof.o …
    ⊢ And (LT.lt ((↑l).toFinset.sup fun a => Order.succ (Profinite.NobelingProof.o …
  -/
  refine ⟨hslt, fun he ↦ h ?_⟩
  have hlt : ∀ i ∈ l.val, ord I i < Finset.sup l.val.toFinset (fun a ↦ Order.succ (ord I a)) := by
    intro i hi
    simp only [Finset.lt_sup_iff, List.mem_toFinset, Order.lt_succ_iff]
    exact ⟨i, hi, le_rfl⟩
  rwa [eval_πs_image' C (le_of_lt hslt) hlt, ← eval_πs' C (le_of_lt hslt) hlt,
    Submodule.apply_mem_span_image_iff_mem_span (injective_πs' C _)]


theorem GoodProducts.union : range C = ⋃ (e : {o' // o' < o}), (smaller C e.val) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    hsC : Profinite.NobelingProof.contained C o
    ⊢ Eq (Profinite.NobelingProof.GoodProducts.range C) (Set.iUnion fun e => Profi …
  -/
  ext p
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    hsC : Profinite.NobelingProof.contained C o
    p : LocallyConstant (↑C) Int
    ⊢ Iff (Membership.mem (Profinite.NobelingProof.GoodProducts.range C) p) (Membe …
  -/
  simp only [smaller, range, Set.mem_iUnion, Set.mem_image, Set.mem_range, Subtype.exists]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    hsC : Profinite.NobelingProof.contained C o
    p : LocallyConstant (↑C) Int
    ⊢ Iff (Exists fun a => Exists fun b => Eq (Profinite.NobelingProof.GoodProduct …
  -/
  refine ⟨fun hp ↦ ?_, fun hp ↦ ?_⟩
    /-
      case h.refine_1
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      p : LocallyConstant (↑C) Int
      hp : Exists fun a => Exists fun b => Eq (Profinite.NobelingProof.GoodProducts. …
      ⊢ Exists fun a => Exists fun h => Exists fun x => And (Exists fun a_1 => Exist …
    -/
  · obtain ⟨l, hl, rfl⟩ := hp
    /-
      case h.refine_1.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood C l
      ⊢ Exists fun a => Exists fun h => Exists fun x => And (Exists fun a_1 => Exist …
    -/
    rw [contained_eq_proj C o hsC, Products.limitOrdinal C ho] at hl
    /-
      case h.refine_1.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      l : Profinite.NobelingProof.Products I
      hl✝ : Profinite.NobelingProof.Products.isGood C l
      hl : Exists fun o' => And (LT.lt o' o) (Profinite.NobelingProof.Products.isGoo …
      ⊢ Exists fun a => Exists fun h => Exists fun x => And (Exists fun a_1 => Exist …
    -/
    obtain ⟨o', ho'⟩ := hl
    /-
      case h.refine_1.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood C l
      o' : Ordinal.{u}
      ho' : And (LT.lt o' o) (Profinite.NobelingProof.Products.isGood (Profinite.Nob …
      ⊢ Exists fun a => Exists fun h => Exists fun x => And (Exists fun a_1 => Exist …
    -/
    refine ⟨o', ho'.1, eval (π C (ord I · < o')) ⟨l, ho'.2⟩, ⟨l, ho'.2, rfl⟩, ?_⟩
    /-
      case h.refine_1.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood C l
      o' : Ordinal.{u}
      ho' : And (LT.lt o' o) (Profinite.NobelingProof.Products.isGood (Profinite.Nob …
      ⊢ Eq ((Profinite.NobelingProof.πs C o') (Profinite.NobelingProof.GoodProducts. …
    -/
    exact Products.eval_πs C (Products.prop_of_isGood  C _ ho'.2)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      p : LocallyConstant (↑C) Int
      hp : Exists fun a => Exists fun h => Exists fun x => And (Exists fun a_1 => Ex …
      ⊢ Exists fun a => Exists fun b => Eq (Profinite.NobelingProof.GoodProducts.eva …
    -/
  · obtain ⟨o', h, _, ⟨l, hl, rfl⟩, rfl⟩ := hp
    /-
      case h.refine_2.intro.intro.intro.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      o' : Ordinal.{u}
      h : LT.lt o' o
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Exists fun a => Exists fun b => Eq (Profinite.NobelingProof.GoodProducts.eva …
    -/
    refine ⟨l, ?_, (Products.eval_πs C (Products.prop_of_isGood  C _ hl)).symm⟩
    /-
      case h.refine_2.intro.intro.intro.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      o' : Ordinal.{u}
      h : LT.lt o' o
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Profinite.NobelingProof.Products.isGood C l
    -/
    rw [contained_eq_proj C o hsC]
    /-
      case h.refine_2.intro.intro.intro.intro.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : o.IsLimit
      hsC : Profinite.NobelingProof.contained C o
      o' : Ordinal.{u}
      h : LT.lt o' o
      l : Profinite.NobelingProof.Products I
      hl : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun x = …
    -/
    exact Products.isGood_mono C (le_of_lt h) hl
    /-
      🎉 no goals
    -/


/--
The image of the `GoodProducts` in `C` is equivalent to the union of `smaller C o'` over all
ordinals `o' < o`.
-/
def GoodProducts.range_equiv : range C ≃ ⋃ (e : {o' // o' < o}), (smaller C e.val) :=
  Equiv.Set.ofEq (union C ho hsC)


theorem GoodProducts.range_equiv_factorization :
    (fun (p : ⋃ (e : {o' // o' < o}), (smaller C e.val)) ↦ p.1) ∘ (range_equiv C ho hsC).toFun =
    (fun (p : range C) ↦ (p.1 : LocallyConstant C ℤ)) := rfl


theorem GoodProducts.linearIndependent_iff_union_smaller :
    LinearIndependent ℤ (GoodProducts.eval C) ↔
      LinearIndependent ℤ (fun (p : ⋃ (e : {o' // o' < o}), (smaller C e.val)) ↦ p.1) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    hsC : Profinite.NobelingProof.contained C o
    ⊢ Iff (LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)) (L …
  -/
  rw [GoodProducts.linearIndependent_iff_range, ← range_equiv_factorization C ho hsC]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    hsC : Profinite.NobelingProof.contained C o
    ⊢ Iff (LinearIndependent Int (Function.comp (fun p => ↑p) (Profinite.NobelingP …
  -/
  exact linearIndependent_equiv (range_equiv C ho hsC)
  /-
    🎉 no goals
  -/


/-- The subset of `C` consisting of those elements whose `o`-th entry is `false`. -/
def C0 := C ∩ {f | f (term I ho) = false}


/-- The subset of `C` consisting of those elements whose `o`-th entry is `true`. -/
def C1 := C ∩ {f | f (term I ho) = true}


include hC in
theorem isClosed_C0 : IsClosed (C0 C ho) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ IsClosed (Profinite.NobelingProof.C0 C ho)
  -/
  refine hC.inter ?_
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ IsClosed (setOf fun f => Eq (f (Profinite.NobelingProof.term I ho)) Bool.fal …
  -/
  have h : Continuous (fun (f : I → Bool) ↦ f (term I ho)) := continuous_apply (term I ho)
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h : Continuous fun f => f (Profinite.NobelingProof.term I ho)
    ⊢ IsClosed (setOf fun f => Eq (f (Profinite.NobelingProof.term I ho)) Bool.fal …
  -/
  exact IsClosed.preimage h (t := {false}) (isClosed_discrete _)
  /-
    🎉 no goals
  -/


include hC in
theorem isClosed_C1 : IsClosed (C1 C ho) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ IsClosed (Profinite.NobelingProof.C1 C ho)
  -/
  refine hC.inter ?_
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ IsClosed (setOf fun f => Eq (f (Profinite.NobelingProof.term I ho)) Bool.true)
  -/
  have h : Continuous (fun (f : I → Bool) ↦ f (term I ho)) := continuous_apply (term I ho)
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h : Continuous fun f => f (Profinite.NobelingProof.term I ho)
    ⊢ IsClosed (setOf fun f => Eq (f (Profinite.NobelingProof.term I ho)) Bool.true)
  -/
  exact IsClosed.preimage h (t := {true}) (isClosed_discrete _)
  /-
    🎉 no goals
  -/


theorem contained_C1 : contained (π (C1 C ho) (ord I · < o)) o :=
  contained_proj _ _


theorem union_C0C1_eq : (C0 C ho) ∪ (C1 C ho) = C := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Union.union (Profinite.NobelingProof.C0 C ho) (Profinite.NobelingProof.C …
  -/
  ext x
  simp only [C0, C1, Set.mem_union, Set.mem_inter_iff, Set.mem_setOf_eq,
    ← and_or_left, and_iff_left_iff_imp, Bool.dichotomy (x (term I ho)), implies_true]


/--
The intersection of `C0` and the projection of `C1`. We will apply the inductive hypothesis to
this set.
-/
def C' := C0 C ho ∩ π (C1 C ho) (ord I · < o)


include hC in
theorem isClosed_C' : IsClosed (C' C ho) :=
  IsClosed.inter (isClosed_C0 _ hC _) (isClosed_proj _ _ (isClosed_C1 _ hC _))


theorem contained_C' : contained (C' C ho) o := fun f hf i hi ↦ contained_C1 C ho f hf.2 i hi


/-- Swapping the `o`-th coordinate to `true`. -/
noncomputable
def SwapTrue : (I → Bool) → I → Bool :=
  fun f i ↦ if ord I i = o then true else f i


theorem continuous_swapTrue  :
    Continuous (SwapTrue o : (I → Bool) → I → Bool) := by
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Continuous (Profinite.NobelingProof.SwapTrue o)
  -/
  dsimp (config := { unfoldPartialApp := true }) [SwapTrue]
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Continuous fun f i => ite (Eq (Profinite.NobelingProof.ord I i) o) Bool.true …
  -/
  apply continuous_pi
  /-
    case h
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ ∀ (i : I), Continuous fun a => ite (Eq (Profinite.NobelingProof.ord I i) o)  …
  -/
  intro i
  /-
    case h
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    i : I
    ⊢ Continuous fun a => ite (Eq (Profinite.NobelingProof.ord I i) o) Bool.true ( …
  -/
  apply Continuous.comp'
    /-
      case h.hg
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      i : I
      ⊢ Continuous (ite (Eq (Profinite.NobelingProof.ord I i) o) Bool.true)
    -/
  · apply continuous_bot
    /-
      🎉 no goals
    -/
    /-
      case h.hf
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      i : I
      ⊢ Continuous fun x => x i
    -/
  · apply continuous_apply
    /-
      🎉 no goals
    -/


include hsC in
theorem swapTrue_mem_C1 (f : π (C1 C ho) (ord I · < o)) :
    SwapTrue o f.val ∈ C1 C ho := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    f : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => LT. …
    ⊢ Membership.mem (Profinite.NobelingProof.C1 C ho) (Profinite.NobelingProof.Sw …
  -/
  obtain ⟨f, g, hg, rfl⟩ := f
  /-
    case mk.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    g : I → Bool
    hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
    ⊢ Membership.mem (Profinite.NobelingProof.C1 C ho) (Profinite.NobelingProof.Sw …
  -/
  convert hg
  /-
    case h.e'_5
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    g : I → Bool
    hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
    ⊢ Eq (Profinite.NobelingProof.SwapTrue o ↑⟨Profinite.NobelingProof.Proj (fun x …
  -/
  dsimp (config := { unfoldPartialApp := true }) [SwapTrue]
  /-
    case h.e'_5
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    g : I → Bool
    hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
    ⊢ Eq (fun i => ite (Eq (Profinite.NobelingProof.ord I i) o) Bool.true (Profini …
  -/
  ext i
  /-
    case h.e'_5.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    g : I → Bool
    hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
    i : I
    ⊢ Eq (ite (Eq (Profinite.NobelingProof.ord I i) o) Bool.true (Profinite.Nobeli …
  -/
  split_ifs with h
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Eq (Profinite.NobelingProof.ord I i) o
      ⊢ Eq Bool.true (g i)
    -/
  · rw [ord_term ho] at h
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Eq (Profinite.NobelingProof.term I ho) i
      ⊢ Eq Bool.true (g i)
    -/
    simpa only [← h] using hg.2.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Not (Eq (Profinite.NobelingProof.ord I i) o)
      ⊢ Eq (Profinite.NobelingProof.Proj (fun x => LT.lt (Profinite.NobelingProof.or …
    -/
  · simp only [Proj, ite_eq_left_iff, not_lt, @eq_comm _ false, ← Bool.not_eq_true]
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Not (Eq (Profinite.NobelingProof.ord I i) o)
      ⊢ LE.le o (Profinite.NobelingProof.ord I i) → Not (Eq (g i) Bool.true)
    -/
    specialize hsC g hg.1 i
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Not (Eq (Profinite.NobelingProof.ord I i) o)
      hsC : Eq (g i) Bool.true → LT.lt (Profinite.NobelingProof.ord I i) (Order.succ …
      ⊢ LE.le o (Profinite.NobelingProof.ord I i) → Not (Eq (g i) Bool.true)
    -/
    intro h'
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Not (Eq (Profinite.NobelingProof.ord I i) o)
      hsC : Eq (g i) Bool.true → LT.lt (Profinite.NobelingProof.ord I i) (Order.succ …
      h' : LE.le o (Profinite.NobelingProof.ord I i)
      ⊢ Not (Eq (g i) Bool.true)
    -/
    contrapose! hsC
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      g : I → Bool
      hg : Membership.mem (Profinite.NobelingProof.C1 C ho) g
      i : I
      h : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h' : LE.le o (Profinite.NobelingProof.ord I i)
      hsC : Eq (g i) Bool.true
      ⊢ And (Eq (g i) Bool.true) (LE.le (Order.succ o) (Profinite.NobelingProof.ord  …
    -/
    exact ⟨hsC, Order.succ_le_of_lt (h'.lt_of_ne' h)⟩
    /-
      🎉 no goals
    -/


/-- The first way to map `C'` into `C`. -/
def CC'₀ : C' C ho → C := fun g ↦ ⟨g.val,g.prop.1.1⟩


/-- The second way to map `C'` into `C`. -/
noncomputable
def CC'₁ : C' C ho → C :=
  fun g ↦ ⟨SwapTrue o g.val, (swapTrue_mem_C1 C hsC ho ⟨g.val,g.prop.2⟩).1⟩


theorem continuous_CC'₀ : Continuous (CC'₀ C ho) := Continuous.subtype_mk continuous_subtype_val _


theorem continuous_CC'₁ : Continuous (CC'₁ C hsC ho) :=
  Continuous.subtype_mk (Continuous.comp (continuous_swapTrue o) continuous_subtype_val) _


/-- The `ℤ`-linear map induced by precomposing with `CC'₀` -/
noncomputable
def Linear_CC'₀ : LocallyConstant C ℤ →ₗ[ℤ] LocallyConstant (C' C ho) ℤ :=
  LocallyConstant.comapₗ ℤ ⟨(CC'₀ C ho), (continuous_CC'₀ C ho)⟩


/-- The `ℤ`-linear map induced by precomposing with `CC'₁` -/
noncomputable
def Linear_CC'₁ : LocallyConstant C ℤ →ₗ[ℤ] LocallyConstant (C' C ho) ℤ :=
  LocallyConstant.comapₗ ℤ ⟨(CC'₁ C hsC ho), (continuous_CC'₁ C hsC ho)⟩


/-- The difference between `Linear_CC'₁` and `Linear_CC'₀`. -/
noncomputable
def Linear_CC' : LocallyConstant C ℤ →ₗ[ℤ] LocallyConstant (C' C ho) ℤ :=
  Linear_CC'₁ C hsC ho - Linear_CC'₀ C ho


theorem CC_comp_zero : ∀ y, (Linear_CC' C hsC ho) ((πs C o) y) = 0 := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ ∀ (y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profin …
  -/
  intro y
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    ⊢ Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) ((Profinite.NobelingProof. …
  -/
  ext x
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    ⊢ Eq (((Profinite.NobelingProof.Linear_CC' C hsC ho) ((Profinite.NobelingProof …
  -/
  dsimp [Linear_CC', Linear_CC'₀, Linear_CC'₁, LocallyConstant.sub_apply]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    ⊢ Eq (HSub.hSub (y (Profinite.NobelingProof.ProjRestrict C (fun x => LT.lt (Pr …
  -/
  simp only [Pi.zero_apply, sub_eq_zero]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    ⊢ Eq (y (Profinite.NobelingProof.ProjRestrict C (fun x => LT.lt (Profinite.Nob …
  -/
  congr 1
  /-
    case h.h.e_6.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    ⊢ Eq (Profinite.NobelingProof.ProjRestrict C (fun x => LT.lt (Profinite.Nobeli …
  -/
  ext i
  /-
    case h.h.e_6.h.a.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    i : I
    ⊢ Eq (↑(Profinite.NobelingProof.ProjRestrict C (fun x => LT.lt (Profinite.Nobe …
  -/
  dsimp [CC'₀, CC'₁, ProjRestrict, Proj]
  /-
    case h.h.e_6.h.a.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    i : I
    ⊢ Eq (ite (LT.lt (Profinite.NobelingProof.ord I i) o) (Profinite.NobelingProof …
  -/
  apply if_ctx_congr Iff.rfl _ (fun _ ↦ rfl)
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    i : I
    ⊢ LT.lt (Profinite.NobelingProof.ord I i) o → Eq (Profinite.NobelingProof.Swap …
  -/
  simp only [SwapTrue, ite_eq_right_iff]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    i : I
    ⊢ LT.lt (Profinite.NobelingProof.ord I i) o → Eq (Profinite.NobelingProof.ord  …
  -/
  intro h₁ h₂
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    x : ↑(Profinite.NobelingProof.C' C ho)
    i : I
    h₁ : LT.lt (Profinite.NobelingProof.ord I i) o
    h₂ : Eq (Profinite.NobelingProof.ord I i) o
    ⊢ Eq Bool.true (↑x i)
  -/
  exact (h₁.ne h₂).elim
  /-
    🎉 no goals
  -/


include hsC in
theorem C0_projOrd {x : I → Bool} (hx : x ∈ C0 C ho) : Proj (ord I · < o) x = x := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
    ⊢ Eq (Profinite.NobelingProof.Proj (fun x => LT.lt (Profinite.NobelingProof.or …
  -/
  ext i
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
    i : I
    ⊢ Eq (Profinite.NobelingProof.Proj (fun x => LT.lt (Profinite.NobelingProof.or …
  -/
  simp only [Proj, Set.mem_setOf, ite_eq_left_iff, not_lt]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
    i : I
    ⊢ LE.le o (Profinite.NobelingProof.ord I i) → Eq Bool.false (x i)
  -/
  intro hi
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
    i : I
    hi : LE.le o (Profinite.NobelingProof.ord I i)
    ⊢ Eq Bool.false (x i)
  -/
  rw [le_iff_lt_or_eq] at hi
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
    i : I
    hi : Or (LT.lt o (Profinite.NobelingProof.ord I i)) (Eq o (Profinite.NobelingP …
    ⊢ Eq Bool.false (x i)
  -/
  cases' hi with hi hi
    /-
      case h.inl
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
      i : I
      hi : LT.lt o (Profinite.NobelingProof.ord I i)
      ⊢ Eq Bool.false (x i)
    -/
  · specialize hsC x hx.1 i
    /-
      case h.inl
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
      i : I
      hi : LT.lt o (Profinite.NobelingProof.ord I i)
      hsC : Eq (x i) Bool.true → LT.lt (Profinite.NobelingProof.ord I i) (Order.succ …
      ⊢ Eq Bool.false (x i)
    -/
    rw [← not_imp_not] at hsC
    /-
      case h.inl
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
      i : I
      hi : LT.lt o (Profinite.NobelingProof.ord I i)
      hsC : Not (LT.lt (Profinite.NobelingProof.ord I i) (Order.succ o)) → Not (Eq ( …
      ⊢ Eq Bool.false (x i)
    -/
    simp only [not_lt, Bool.not_eq_true, Order.succ_le_iff] at hsC
    /-
      case h.inl
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
      i : I
      hi : LT.lt o (Profinite.NobelingProof.ord I i)
      hsC : LT.lt o (Profinite.NobelingProof.ord I i) → Eq (x i) Bool.false
      ⊢ Eq Bool.false (x i)
    -/
    exact (hsC hi).symm
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C0 C ho) x
      i : I
      hi : Eq o (Profinite.NobelingProof.ord I i)
      ⊢ Eq Bool.false (x i)
    -/
  · simp only [C0, Set.mem_inter_iff, Set.mem_setOf_eq] at hx
    /-
      case h.inr
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      i : I
      hi : Eq o (Profinite.NobelingProof.ord I i)
      hx : And (Membership.mem C x) (Eq (x (Profinite.NobelingProof.term I ho)) Bool …
      ⊢ Eq Bool.false (x i)
    -/
    rw [eq_comm, ord_term ho] at hi
    /-
      case h.inr
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      i : I
      hi : Eq (Profinite.NobelingProof.term I ho) i
      hx : And (Membership.mem C x) (Eq (x (Profinite.NobelingProof.term I ho)) Bool …
      ⊢ Eq Bool.false (x i)
    -/
    rw [← hx.2, hi]
    /-
      🎉 no goals
    -/


include hsC in
theorem C1_projOrd {x : I → Bool} (hx : x ∈ C1 C ho) : SwapTrue o (Proj (ord I · < o) x) = x := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
    ⊢ Eq (Profinite.NobelingProof.SwapTrue o (Profinite.NobelingProof.Proj (fun x  …
  -/
  ext i
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
    i : I
    ⊢ Eq (Profinite.NobelingProof.SwapTrue o (Profinite.NobelingProof.Proj (fun x  …
  -/
  dsimp [SwapTrue, Proj]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
    i : I
    ⊢ Eq (ite (Eq (Profinite.NobelingProof.ord I i) o) Bool.true (ite (LT.lt (Prof …
  -/
  split_ifs with hi h
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Eq (Profinite.NobelingProof.ord I i) o
      ⊢ Eq Bool.true (x i)
    -/
  · rw [ord_term ho] at hi
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Eq (Profinite.NobelingProof.term I ho) i
      ⊢ Eq Bool.true (x i)
    -/
    rw [← hx.2, hi]
    /-
      🎉 no goals
    -/
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : LT.lt (Profinite.NobelingProof.ord I i) o
      ⊢ Eq (x i) (x i)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : Not (LT.lt (Profinite.NobelingProof.ord I i) o)
      ⊢ Eq Bool.false (x i)
    -/
  · simp only [not_lt] at h
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : LE.le o (Profinite.NobelingProof.ord I i)
      ⊢ Eq Bool.false (x i)
    -/
    have h' : o < ord I i := lt_of_le_of_ne h (Ne.symm hi)
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : LE.le o (Profinite.NobelingProof.ord I i)
      h' : LT.lt o (Profinite.NobelingProof.ord I i)
      ⊢ Eq Bool.false (x i)
    -/
    specialize hsC x hx.1 i
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : LE.le o (Profinite.NobelingProof.ord I i)
      h' : LT.lt o (Profinite.NobelingProof.ord I i)
      hsC : Eq (x i) Bool.true → LT.lt (Profinite.NobelingProof.ord I i) (Order.succ …
      ⊢ Eq Bool.false (x i)
    -/
    rw [← not_imp_not] at hsC
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : LE.le o (Profinite.NobelingProof.ord I i)
      h' : LT.lt o (Profinite.NobelingProof.ord I i)
      hsC : Not (LT.lt (Profinite.NobelingProof.ord I i) (Order.succ o)) → Not (Eq ( …
      ⊢ Eq Bool.false (x i)
    -/
    simp only [not_lt, Bool.not_eq_true, Order.succ_le_iff] at hsC
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x : I → Bool
      hx : Membership.mem (Profinite.NobelingProof.C1 C ho) x
      i : I
      hi : Not (Eq (Profinite.NobelingProof.ord I i) o)
      h : LE.le o (Profinite.NobelingProof.ord I i)
      h' : LT.lt o (Profinite.NobelingProof.ord I i)
      hsC : LT.lt o (Profinite.NobelingProof.ord I i) → Eq (x i) Bool.false
      ⊢ Eq Bool.false (x i)
    -/
    exact (hsC h').symm
    /-
      🎉 no goals
    -/


include hC in
open scoped Classical in
theorem CC_exact {f : LocallyConstant C ℤ} (hf : Linear_CC' C hsC ho f = 0) :
    ∃ y, πs C o y = f := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    f : LocallyConstant (↑C) Int
    hf : Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) f) 0
    ⊢ Exists fun y => Eq ((Profinite.NobelingProof.πs C o) y) f
  -/
  dsimp [Linear_CC', Linear_CC'₀, Linear_CC'₁] at hf
  simp only [sub_eq_zero, ← LocallyConstant.coe_inj, LocallyConstant.coe_comap,
    continuous_CC'₀, continuous_CC'₁] at hf
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    f : LocallyConstant (↑C) Int
    hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
    ⊢ Exists fun y => Eq ((Profinite.NobelingProof.πs C o) y) f
  -/
  let C₀C : C0 C ho → C := fun x ↦ ⟨x.val, x.prop.1⟩
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    f : LocallyConstant (↑C) Int
    hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
    C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
    ⊢ Exists fun y => Eq ((Profinite.NobelingProof.πs C o) y) f
  -/
  have h₀ : Continuous C₀C := Continuous.subtype_mk continuous_induced_dom _
  let C₁C : π (C1 C ho) (ord I · < o) → C :=
    fun x ↦ ⟨SwapTrue o x.val, (swapTrue_mem_C1 C hsC ho x).1⟩
  have h₁ : Continuous C₁C := Continuous.subtype_mk
    ((continuous_swapTrue o).comp continuous_subtype_val) _
  refine ⟨LocallyConstant.piecewise' ?_ (isClosed_C0 C hC ho)
      (isClosed_proj _ o (isClosed_C1 C hC ho)) (f.comap ⟨C₀C, h₀⟩) (f.comap ⟨C₁C, h₁⟩) ?_, ?_⟩
    /-
      case refine_1
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      ⊢ HasSubset.Subset (Profinite.NobelingProof.π C fun x => LT.lt (Profinite.Nobe …
    -/
  · rintro _ ⟨y, hyC, rfl⟩
    /-
      case refine_1.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      y : I → Bool
      hyC : Membership.mem C y
      ⊢ Membership.mem (Union.union (Profinite.NobelingProof.C0 C ho) (Profinite.Nob …
    -/
    simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    /-
      case refine_1.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      y : I → Bool
      hyC : Membership.mem C y
      ⊢ Or (Membership.mem (Profinite.NobelingProof.C0 C ho) (Profinite.NobelingProo …
    -/
    rw [← union_C0C1_eq C ho] at hyC
    /-
      case refine_1.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      y : I → Bool
      hyC : Membership.mem (Union.union (Profinite.NobelingProof.C0 C ho) (Profinite …
      ⊢ Or (Membership.mem (Profinite.NobelingProof.C0 C ho) (Profinite.NobelingProo …
    -/
    refine hyC.imp (fun hyC ↦ ?_) (fun hyC ↦ ⟨y, hyC, rfl⟩)
    /-
      case refine_1.intro.intro
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      y : I → Bool
      hyC✝ : Membership.mem (Union.union (Profinite.NobelingProof.C0 C ho) (Profinit …
      hyC : Membership.mem (Profinite.NobelingProof.C0 C ho) y
      ⊢ Membership.mem (Profinite.NobelingProof.C0 C ho) (Profinite.NobelingProof.Pr …
    -/
    rwa [C0_projOrd C hsC ho hyC]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      ⊢ ∀ (x : I → Bool) (hx : Membership.mem (Inter.inter (Profinite.NobelingProof. …
    -/
  · intro x hx
    /-
      case refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      x : I → Bool
      hx : Membership.mem (Inter.inter (Profinite.NobelingProof.C0 C ho) (Profinite. …
      ⊢ Eq ((LocallyConstant.comap { toFun := C₀C, continuous_toFun := h₀ } f) ⟨x, ⋯ …
    -/
    simpa only [h₀, h₁, LocallyConstant.coe_comap] using (congrFun hf ⟨x, hx⟩).symm
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      ⊢ Eq ((Profinite.NobelingProof.πs C o) (LocallyConstant.piecewise' ⋯ ⋯ ⋯ (Loca …
    -/
  · ext ⟨x, hx⟩
    /-
      case refine_3.h.mk
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      x : I → Bool
      hx : Membership.mem C x
      ⊢ Eq (((Profinite.NobelingProof.πs C o) (LocallyConstant.piecewise' ⋯ ⋯ ⋯ (Loc …
    -/
    rw [← union_C0C1_eq C ho] at hx
    /-
      case refine_3.h.mk
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      f : LocallyConstant (↑C) Int
      hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
      C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
      h₀ : Continuous C₀C
      C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
      h₁ : Continuous C₁C
      x : I → Bool
      hx✝ : Membership.mem C x
      hx : Membership.mem (Union.union (Profinite.NobelingProof.C0 C ho) (Profinite. …
      ⊢ Eq (((Profinite.NobelingProof.πs C o) (LocallyConstant.piecewise' ⋯ ⋯ ⋯ (Loc …
    -/
    cases' hx with hx₀ hx₁
    · have hx₀' : ProjRestrict C (ord I · < o) ⟨x, hx⟩ = x := by
        simpa only [ProjRestrict, Set.MapsTo.val_restrict_apply] using C0_projOrd C hsC ho hx₀
      simp only [C₀C, πs_apply_apply, hx₀', hx₀, LocallyConstant.piecewise'_apply_left,
        LocallyConstant.coe_comap, ContinuousMap.coe_mk, Function.comp_apply]
    · have hx₁' : (ProjRestrict C (ord I · < o) ⟨x, hx⟩).val ∈ π (C1 C ho) (ord I · < o) := by
        simpa only [ProjRestrict, Set.MapsTo.val_restrict_apply] using ⟨x, hx₁, rfl⟩
      simp only [C₁C, πs_apply_apply, continuous_projRestrict, LocallyConstant.coe_comap,
        Function.comp_apply, hx₁', LocallyConstant.piecewise'_apply_right, h₁]
      /-
        case refine_3.h.mk.inr
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hC : IsClosed C
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        f : LocallyConstant (↑C) Int
        hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
        C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
        h₀ : Continuous C₀C
        C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
        h₁ : Continuous C₁C
        x : I → Bool
        hx : Membership.mem C x
        hx₁ : Membership.mem (Profinite.NobelingProof.C1 C ho) x
        hx₁' : Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C …
        ⊢ Eq (f ({ toFun := fun x => ⟨Profinite.NobelingProof.SwapTrue o ↑x, ⋯⟩, conti …
      -/
      congr
      /-
        case refine_3.h.mk.inr.h.e_6.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hC : IsClosed C
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        f : LocallyConstant (↑C) Int
        hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
        C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
        h₀ : Continuous C₀C
        C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
        h₁ : Continuous C₁C
        x : I → Bool
        hx : Membership.mem C x
        hx₁ : Membership.mem (Profinite.NobelingProof.C1 C ho) x
        hx₁' : Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C …
        ⊢ Eq ({ toFun := fun x => ⟨Profinite.NobelingProof.SwapTrue o ↑x, ⋯⟩, continuo …
      -/
      simp only [ContinuousMap.coe_mk, Subtype.mk.injEq]
      /-
        case refine_3.h.mk.inr.h.e_6.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hC : IsClosed C
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        f : LocallyConstant (↑C) Int
        hf : Eq ⇑((LocallyConstant.comapₗ Int { toFun := Profinite.NobelingProof.CC'₁  …
        C₀C : ↑(Profinite.NobelingProof.C0 C ho) → ↑C := fun x => ⟨↑x, ⋯⟩
        h₀ : Continuous C₀C
        C₁C : ↑(Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C ho) fun x => L …
        h₁ : Continuous C₁C
        x : I → Bool
        hx : Membership.mem C x
        hx₁ : Membership.mem (Profinite.NobelingProof.C1 C ho) x
        hx₁' : Membership.mem (Profinite.NobelingProof.π (Profinite.NobelingProof.C1 C …
        ⊢ Eq (Profinite.NobelingProof.SwapTrue o ↑(Profinite.NobelingProof.ProjRestric …
      -/
      exact C1_projOrd C hsC ho hx₁
      /-
        🎉 no goals
      -/


variable (o) in
theorem succ_mono : CategoryTheory.Mono (ModuleCat.ofHom (πs C o)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ CategoryTheory.Mono (ModuleCat.ofHom (Profinite.NobelingProof.πs C o))
  -/
  rw [ModuleCat.mono_iff_injective]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ⊢ Function.Injective ⇑(ModuleCat.ofHom (Profinite.NobelingProof.πs C o)).hom
  -/
  exact injective_πs _ _
  /-
    🎉 no goals
  -/


include hC in
theorem succ_exact :
    (ShortComplex.mk (ModuleCat.ofHom (πs C o)) (ModuleCat.ofHom (Linear_CC' C hsC ho))
        /-
          I : Type u
          C : Set (I → Bool)
          inst✝¹ : LinearOrder I
          inst✝ : WellFoundedLT I
          o : Ordinal.{u}
          hC : IsClosed C
          hsC : Profinite.NobelingProof.contained C (Order.succ o)
          ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom (Profinite.NobelingP …
        -/
    (by ext : 2; apply CC_comp_zero)).Exact := by
                 /-
                   🎉 no goals
                 -/
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ (CategoryTheory.ShortComplex.mk (ModuleCat.ofHom (Profinite.NobelingProof.πs …
  -/
  rw [ShortComplex.moduleCat_exact_iff]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ ∀ (x₂ : ↑(CategoryTheory.ShortComplex.mk (ModuleCat.ofHom (Profinite.Nobelin …
  -/
  intro f
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    f : ↑(CategoryTheory.ShortComplex.mk (ModuleCat.ofHom (Profinite.NobelingProof …
    ⊢ Eq ((CategoryTheory.ShortComplex.mk (ModuleCat.ofHom (Profinite.NobelingProo …
  -/
  exact CC_exact C hC hsC ho
  /-
    🎉 no goals
  -/


/--
The `GoodProducts` in `C` that contain `o` (they necessarily start with `o`, see
`GoodProducts.head!_eq_o_of_maxProducts`)
-/
def MaxProducts : Set (Products I) := {l | l.isGood C ∧ term I ho ∈ l.val}


include hsC in
theorem union_succ : GoodProducts C = GoodProducts (π C (ord I · < o)) ∪ MaxProducts C ho := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Profinite.NobelingProof.GoodProducts C) (Union.union (Profinite.Nobeling …
  -/
  ext l
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : Profinite.NobelingProof.Products I
    ⊢ Iff (Membership.mem (Profinite.NobelingProof.GoodProducts C) l) (Membership. …
  -/
  simp only [GoodProducts, MaxProducts, Set.mem_union, Set.mem_setOf_eq]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : Profinite.NobelingProof.Products I
    ⊢ Iff (Profinite.NobelingProof.Products.isGood C l) (Or (Profinite.NobelingPro …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case h.refine_1
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : Profinite.NobelingProof.Products I
      h : Profinite.NobelingProof.Products.isGood C l
      ⊢ Or (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun …
    -/
  · by_cases hh : term I ho ∈ l.val
      /-
        case pos
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Membership.mem (↑l) (Profinite.NobelingProof.term I ho)
        ⊢ Or (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun …
      -/
    · exact Or.inr ⟨h, hh⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Not (Membership.mem (↑l) (Profinite.NobelingProof.term I ho))
        ⊢ Or (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun …
      -/
    · left
      /-
        case neg.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Not (Membership.mem (↑l) (Profinite.NobelingProof.term I ho))
        ⊢ Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun x = …
      -/
      intro he
      /-
        case neg.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Not (Membership.mem (↑l) (Profinite.NobelingProof.term I ho))
        he : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pr …
        ⊢ False
      -/
      apply h
      /-
        case neg.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Not (Membership.mem (↑l) (Profinite.NobelingProof.term I ho))
        he : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pr …
        ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
      -/
      have h' := Products.prop_of_isGood_of_contained C _ h hsC
      /-
        case neg.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Not (Membership.mem (↑l) (Profinite.NobelingProof.term I ho))
        he : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pr …
        h' : ∀ (i : I), Membership.mem (↑l) i → LT.lt (Profinite.NobelingProof.ord I i …
        ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
      -/
      simp only [Order.lt_succ_iff] at h'
      /-
        case neg.h
        I : Type u
        C : Set (I → Bool)
        inst✝¹ : LinearOrder I
        inst✝ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        l : Profinite.NobelingProof.Products I
        h : Profinite.NobelingProof.Products.isGood C l
        hh : Not (Membership.mem (↑l) (Profinite.NobelingProof.term I ho))
        he : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pr …
        h' : ∀ (i : I), Membership.mem (↑l) i → LE.le (Profinite.NobelingProof.ord I i …
        ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
      -/
      simp only [not_imp_not] at hh
      have hh' : ∀ a ∈ l.val, ord I a < o := by
        intro a ha
        refine (h' a ha).lt_of_ne ?_
        rw [ne_eq, ord_term ho a]
        rintro rfl
        contradiction
      rwa [Products.eval_πs_image C hh', ← Products.eval_πs C hh',
        Submodule.apply_mem_span_image_iff_mem_span (injective_πs _ _)]
    /-
      case h.refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : Profinite.NobelingProof.Products I
      h : Or (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C f …
      ⊢ Profinite.NobelingProof.Products.isGood C l
    -/
  · refine h.elim (fun hh ↦ ?_) And.left
    /-
      case h.refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : Profinite.NobelingProof.Products I
      h : Or (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C f …
      hh : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      ⊢ Profinite.NobelingProof.Products.isGood C l
    -/
    have := Products.isGood_mono C (Order.lt_succ o).le hh
    /-
      case h.refine_2
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : Profinite.NobelingProof.Products I
      h : Or (Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C f …
      hh : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fun  …
      this : Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.π C fu …
      ⊢ Profinite.NobelingProof.Products.isGood C l
    -/
    rwa [contained_eq_proj C (Order.succ o) hsC]
    /-
      🎉 no goals
    -/


/-- The inclusion map from the sum of `GoodProducts (π C (ord I · < o))` and
    `(MaxProducts C ho)` to `Products I`. -/
def sum_to : (GoodProducts (π C (ord I · < o))) ⊕ (MaxProducts C ho) → Products I :=
  Sum.elim Subtype.val Subtype.val


theorem injective_sum_to : Function.Injective (sum_to C ho) := by
  refine Function.Injective.sum_elim Subtype.val_injective Subtype.val_injective
    (fun ⟨a,ha⟩ ⟨b,hb⟩  ↦ (fun (hab : a = b) ↦ ?_))
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x✝¹ : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun  …
    x✝ : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    a : Profinite.NobelingProof.Products I
    ha : Membership.mem (Profinite.NobelingProof.GoodProducts (Profinite.NobelingP …
    b : Profinite.NobelingProof.Products I
    hb : Membership.mem (Profinite.NobelingProof.GoodProducts.MaxProducts C ho) b
    hab : Eq a b
    ⊢ False
  -/
  rw [← hab] at hb
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x✝¹ : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun  …
    x✝ : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    a : Profinite.NobelingProof.Products I
    ha : Membership.mem (Profinite.NobelingProof.GoodProducts (Profinite.NobelingP …
    b : Profinite.NobelingProof.Products I
    hb : Membership.mem (Profinite.NobelingProof.GoodProducts.MaxProducts C ho) a
    hab : Eq a b
    ⊢ False
  -/
  have ha' := Products.prop_of_isGood  C _ ha (term I ho) hb.2
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x✝¹ : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun  …
    x✝ : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    a : Profinite.NobelingProof.Products I
    ha : Membership.mem (Profinite.NobelingProof.GoodProducts (Profinite.NobelingP …
    b : Profinite.NobelingProof.Products I
    hb : Membership.mem (Profinite.NobelingProof.GoodProducts.MaxProducts C ho) a
    hab : Eq a b
    ha' : LT.lt (Profinite.NobelingProof.ord I (Profinite.NobelingProof.term I ho) …
    ⊢ False
  -/
  simp only [ord_term_aux, lt_self_iff_false] at ha'
  /-
    🎉 no goals
  -/


theorem sum_to_range :
    Set.range (sum_to C ho) = GoodProducts (π C (ord I · < o)) ∪ MaxProducts C ho := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Set.range (Profinite.NobelingProof.GoodProducts.sum_to C ho)) (Union.uni …
  -/
  have h : Set.range (sum_to C ho) = _ ∪ _ := Set.Sum.elim_range _ _; rw [h]; congr<;> ext l
    /-
      case e_a.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      h : Eq (Set.range (Profinite.NobelingProof.GoodProducts.sum_to C ho)) (Union.u …
      l : Profinite.NobelingProof.Products I
      ⊢ Iff (Membership.mem (Set.range Subtype.val) l) (Membership.mem (Profinite.No …
    -/
  · exact ⟨fun ⟨m,hm⟩ ↦ by rw [← hm]; exact m.prop, fun hl ↦ ⟨⟨l,hl⟩, rfl⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case e_a.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      h : Eq (Set.range (Profinite.NobelingProof.GoodProducts.sum_to C ho)) (Union.u …
      l : Profinite.NobelingProof.Products I
      ⊢ Iff (Membership.mem (Set.range Subtype.val) l) (Membership.mem (Profinite.No …
    -/
  · exact ⟨fun ⟨m,hm⟩ ↦ by rw [← hm]; exact m.prop, fun hl ↦ ⟨⟨l,hl⟩, rfl⟩⟩
    /-
      🎉 no goals
    -/


/-- The equivalence from the sum of `GoodProducts (π C (ord I · < o))` and
    `(MaxProducts C ho)` to `GoodProducts C`. -/
noncomputable
def sum_equiv (hsC : contained C (Order.succ o)) (ho : o < Ordinal.type (·<· : I → I → Prop)) :
    GoodProducts (π C (ord I · < o)) ⊕ (MaxProducts C ho) ≃ GoodProducts C :=
  calc _ ≃ Set.range (sum_to C ho) := Equiv.ofInjective (sum_to C ho) (injective_sum_to C ho)
                                     /-
                                       I : Type u
                                       C : Set (I → Bool)
                                       inst✝¹ : LinearOrder I
                                       inst✝ : WellFoundedLT I
                                       o : Ordinal.{u}
                                       hC : IsClosed C
                                       hsC✝ : Profinite.NobelingProof.contained C (Order.succ o)
                                       ho✝ : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
                                       hsC : Profinite.NobelingProof.contained C (Order.succ o)
                                       ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
                                       ⊢ Eq (Set.range (Profinite.NobelingProof.GoodProducts.sum_to C ho)) (Profinite …
                                     -/
       _ ≃ _ := Equiv.Set.ofEq <| by rw [sum_to_range C ho, union_succ C hsC ho]
                                     /-
                                       🎉 no goals
                                     -/


theorem sum_equiv_comp_eval_eq_elim : eval C ∘ (sum_equiv C hsC ho).toFun =
    (Sum.elim (fun (l : GoodProducts (π C (ord I · < o))) ↦ Products.eval C l.1)
    (fun (l : MaxProducts C ho) ↦ Products.eval C l.1)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Function.comp (Profinite.NobelingProof.GoodProducts.eval C) (Profinite.N …
  -/
  ext ⟨_,_⟩ <;> [rfl; rfl]
  /-
    🎉 no goals
  -/


/-- Let

`N := LocallyConstant (π C (ord I · < o)) ℤ`

`M := LocallyConstant C ℤ`

`P := LocallyConstant (C' C ho) ℤ`

`ι := GoodProducts (π C (ord I · < o))`

`ι' := GoodProducts (C' C ho')`

`v : ι → N := GoodProducts.eval (π C (ord I · < o))`

Then `SumEval C ho` is the map `u` in the diagram below. It is linearly independent if and only if
`GoodProducts.eval C` is, see `linearIndependent_iff_sum`. The top row is the exact sequence given
by `succ_exact` and `succ_mono`. The left square commutes by `GoodProducts.square_commutes`.
```
0 --→ N --→ M --→  P
      ↑     ↑      ↑
     v|    u|      |
      ι → ι ⊕ ι' ← ι'
```
-/
def SumEval : GoodProducts (π C (ord I · < o)) ⊕ MaxProducts C ho →
    LocallyConstant C ℤ :=
  Sum.elim (fun l ↦ l.1.eval C) (fun l ↦ l.1.eval C)


include hsC in
theorem linearIndependent_iff_sum :
    LinearIndependent ℤ (eval C) ↔ LinearIndependent ℤ (SumEval C ho) := by
  rw [← linearIndependent_equiv (sum_equiv C hsC ho), SumEval,
    ← sum_equiv_comp_eval_eq_elim C hsC ho]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Iff (LinearIndependent Int (Function.comp (Profinite.NobelingProof.GoodProdu …
  -/
  exact Iff.rfl
  /-
    🎉 no goals
  -/


include hsC in
theorem span_sum : Set.range (eval C) = Set.range (Sum.elim
    (fun (l : GoodProducts (π C (ord I · < o))) ↦ Products.eval C l.1)
    (fun (l : MaxProducts C ho) ↦ Products.eval C l.1)) := by
  rw [← sum_equiv_comp_eval_eq_elim C hsC ho, Equiv.toFun_as_coe,
    EquivLike.range_comp (e := sum_equiv C hsC ho)]



theorem square_commutes : SumEval C ho ∘ Sum.inl =
    ModuleCat.ofHom (πs C o) ∘ eval (π C (ord I · < o)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Function.comp (Profinite.NobelingProof.GoodProducts.SumEval C ho) Sum.in …
  -/
  ext l
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
    x✝ : ↑C
    ⊢ Eq ((Function.comp (Profinite.NobelingProof.GoodProducts.SumEval C ho) Sum.i …
  -/
  dsimp [SumEval]
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
    x✝ : ↑C
    ⊢ Eq ((Profinite.NobelingProof.Products.eval C ↑l) x✝) ((Profinite.NobelingPro …
  -/
  rw [← Products.eval_πs C (Products.prop_of_isGood  _ _ l.prop)]
  /-
    case h.h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
    x✝ : ↑C
    ⊢ Eq (((Profinite.NobelingProof.πs C o) (Profinite.NobelingProof.Products.eval …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem swapTrue_eq_true (x : I → Bool) : SwapTrue o x (term I ho) = true := by
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    x : I → Bool
    ⊢ Eq (Profinite.NobelingProof.SwapTrue o x (Profinite.NobelingProof.term I ho) …
  -/
  simp only [SwapTrue, ord_term_aux, ite_true]
  /-
    🎉 no goals
  -/


theorem mem_C'_eq_false : ∀ x, x ∈ C' C ho → x (term I ho) = false := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ ∀ (x : I → Bool), Membership.mem (Profinite.NobelingProof.C' C ho) x → Eq (x …
  -/
  rintro x ⟨_, y, _, rfl⟩
  /-
    case intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    y : I → Bool
    left✝¹ : Membership.mem (Profinite.NobelingProof.C1 C ho) y
    left✝ : Membership.mem (Profinite.NobelingProof.C0 C ho) (Profinite.NobelingPr …
    ⊢ Eq (Profinite.NobelingProof.Proj (fun x => LT.lt (Profinite.NobelingProof.or …
  -/
  simp only [Proj, ord_term_aux, lt_self_iff_false, ite_false]
  /-
    🎉 no goals
  -/


/-- `List.tail` as a `Products`. -/
def Products.Tail (l : Products I) : Products I :=
  ⟨l.val.tail, List.Chain'.tail l.prop⟩


theorem Products.max_eq_o_cons_tail [Inhabited I] (l : Products I) (hl : l.val ≠ [])
    (hlh : l.val.head! = term I ho) : l.val = term I ho :: l.Tail.val := by
  /-
    I : Type u
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    ⊢ Eq (↑l) (List.cons (Profinite.NobelingProof.term I ho) ↑l.Tail)
  -/
  rw [← List.cons_head!_tail hl, hlh]
  /-
    I : Type u
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    ⊢ Eq (List.cons (Profinite.NobelingProof.term I ho) (↑l).tail) (List.cons (Pro …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem Products.max_eq_o_cons_tail' [Inhabited I] (l : Products I) (hl : l.val ≠ [])
    (hlh : l.val.head! = term I ho) (hlc : List.Chain' (·>·) (term I ho :: l.Tail.val)) :
    l = ⟨term I ho :: l.Tail.val, hlc⟩ := by
  /-
    I : Type u
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    ⊢ Eq l ⟨List.cons (Profinite.NobelingProof.term I ho) ↑l.Tail, hlc⟩
  -/
  simp_rw [← max_eq_o_cons_tail ho l hl hlh]
  /-
    I : Type u
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    ⊢ Eq l ⟨↑l, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


include hsC in
theorem GoodProducts.head!_eq_o_of_maxProducts [Inhabited I] (l : ↑(MaxProducts C ho)) :
    l.val.val.head! = term I ho := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    ⊢ Eq (↑↑l).head! (Profinite.NobelingProof.term I ho)
  -/
  rw [eq_comm, ← ord_term ho]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    ⊢ Eq (Profinite.NobelingProof.ord I (↑↑l).head!) o
  -/
  have hm := l.prop.2
  have := Products.prop_of_isGood_of_contained C _ l.prop.1 hsC l.val.val.head!
    (List.head!_mem_self (List.ne_nil_of_mem hm))
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    hm : Membership.mem (↑↑l) (Profinite.NobelingProof.term I ho)
    this : LT.lt (Profinite.NobelingProof.ord I (↑↑l).head!) (Order.succ o)
    ⊢ Eq (Profinite.NobelingProof.ord I (↑↑l).head!) o
  -/
  simp only [Order.lt_succ_iff] at this
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    hm : Membership.mem (↑↑l) (Profinite.NobelingProof.term I ho)
    this : LE.le (Profinite.NobelingProof.ord I (↑↑l).head!) o
    ⊢ Eq (Profinite.NobelingProof.ord I (↑↑l).head!) o
  -/
  refine eq_of_le_of_not_lt this (not_lt.mpr ?_)
  have h : ord I (term I ho) ≤ ord I l.val.val.head! := by
    simp only [← ord_term_aux, ord, Ordinal.typein_le_typein, not_lt]
    exact Products.rel_head!_of_mem hm
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    hm : Membership.mem (↑↑l) (Profinite.NobelingProof.term I ho)
    this : LE.le (Profinite.NobelingProof.ord I (↑↑l).head!) o
    h : LE.le (Profinite.NobelingProof.ord I (Profinite.NobelingProof.term I ho))  …
    ⊢ LE.le o (Profinite.NobelingProof.ord I (↑↑l).head!)
  -/
  rwa [ord_term_aux] at h
  /-
    🎉 no goals
  -/


include hsC in
theorem GoodProducts.max_eq_o_cons_tail (l : MaxProducts C ho) :
    l.val.val = (term I ho) :: l.val.Tail.val :=
  have : Inhabited I := ⟨term I ho⟩
  Products.max_eq_o_cons_tail ho l.val (List.ne_nil_of_mem l.prop.2)
    (head!_eq_o_of_maxProducts _ hsC ho l)


theorem Products.evalCons {I} [LinearOrder I] {C : Set (I → Bool)} {l : List I} {a : I}
    (hla : (a::l).Chain' (·>·)) : Products.eval C ⟨a::l,hla⟩ =
    (e C a) * Products.eval C ⟨l,List.Chain'.sublist hla (List.tail_sublist (a::l))⟩ := by
  /-
    I : Type u_1
    inst✝ : LinearOrder I
    C : Set (I → Bool)
    l : List I
    a : I
    hla : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons a l)
    ⊢ Eq (Profinite.NobelingProof.Products.eval C ⟨List.cons a l, hla⟩) (HMul.hMul …
  -/
  simp only [eval.eq_1, List.map, List.prod_cons]
  /-
    🎉 no goals
  -/


theorem Products.max_eq_eval [Inhabited I] (l : Products I) (hl : l.val ≠ [])
    (hlh : l.val.head! = term I ho) :
    Linear_CC' C hsC ho (l.eval C) = l.Tail.eval (C' C ho) := by
  have hlc : ((term I ho) :: l.Tail.val).Chain' (·>·) := by
    rw [← max_eq_o_cons_tail ho l hl hlh]; exact l.prop
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    ⊢ Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) (Profinite.NobelingProof.P …
  -/
  rw [max_eq_o_cons_tail' ho l hl hlh hlc, Products.evalCons]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    ⊢ Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) (HMul.hMul (Profinite.Nobe …
  -/
  ext x
  simp only [Linear_CC', Linear_CC'₁, LocallyConstant.comapₗ, Linear_CC'₀, Subtype.coe_eta,
    LinearMap.sub_apply, LinearMap.coe_mk, AddHom.coe_mk, LocallyConstant.sub_apply,
    LocallyConstant.coe_comap, LocallyConstant.coe_mul, ContinuousMap.coe_mk, Function.comp_apply,
    Pi.mul_apply]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    x : ↑(Profinite.NobelingProof.C' C ho)
    ⊢ Eq (HSub.hSub (HMul.hMul ((Profinite.NobelingProof.e C (Profinite.NobelingPr …
  -/
  rw [CC'₁, CC'₀, Products.eval_eq, Products.eval_eq, Products.eval_eq]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    x : ↑(Profinite.NobelingProof.C' C ho)
    ⊢ Eq (HSub.hSub (HMul.hMul ((Profinite.NobelingProof.e C (Profinite.NobelingPr …
  -/
  simp only [mul_ite, mul_one, mul_zero]
  have hi' : ∀ i, i ∈ l.Tail.val → (x.val i = SwapTrue o x.val i) := by
    intro i hi
    simp only [SwapTrue, @eq_comm _ (x.val i), ite_eq_right_iff, ord_term ho]
    rintro rfl
    exact ((List.Chain.rel hlc hi).ne rfl).elim
  have H : (∀ i, i ∈ l.Tail.val → (x.val i = true)) =
      (∀ i, i ∈ l.Tail.val → (SwapTrue o x.val i = true)) := by
    apply forall_congr; intro i; apply forall_congr; intro hi; rw [hi' i hi]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    x : ↑(Profinite.NobelingProof.C' C ho)
    hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
    H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
    ⊢ Eq (HSub.hSub (ite (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.No …
  -/
  simp only [H]
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝² : LinearOrder I
    inst✝¹ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    inst✝ : Inhabited I
    l : Profinite.NobelingProof.Products I
    hl : Ne (↑l) List.nil
    hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
    hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
    x : ↑(Profinite.NobelingProof.C' C ho)
    hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
    H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
    ⊢ Eq (HSub.hSub (ite (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.No …
  -/
  split_ifs with h₁ h₂ h₃ <;> try (dsimp [e])
                              /-
                                🎉 no goals
                              -/
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝² : LinearOrder I
      inst✝¹ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      inst✝ : Inhabited I
      l : Profinite.NobelingProof.Products I
      hl : Ne (↑l) List.nil
      hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
      hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
      x : ↑(Profinite.NobelingProof.C' C ho)
      hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
      H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
      h₁ : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.NobelingProof.SwapT …
      h₂ : ∀ (i : I), Membership.mem (↑(Profinite.NobelingProof.Products.Tail ⟨List. …
      ⊢ Eq (HSub.hSub (ite (Eq (Profinite.NobelingProof.SwapTrue o (↑x) (Profinite.N …
    -/
  · rw [if_pos (swapTrue_eq_true _ _), if_neg]
      /-
        case pos
        I : Type u
        C : Set (I → Bool)
        inst✝² : LinearOrder I
        inst✝¹ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        inst✝ : Inhabited I
        l : Profinite.NobelingProof.Products I
        hl : Ne (↑l) List.nil
        hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
        hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
        x : ↑(Profinite.NobelingProof.C' C ho)
        hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
        H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
        h₁ : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.NobelingProof.SwapT …
        h₂ : ∀ (i : I), Membership.mem (↑(Profinite.NobelingProof.Products.Tail ⟨List. …
        ⊢ Eq (HSub.hSub 1 0) 1
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos.hnc
        I : Type u
        C : Set (I → Bool)
        inst✝² : LinearOrder I
        inst✝¹ : WellFoundedLT I
        o : Ordinal.{u}
        hsC : Profinite.NobelingProof.contained C (Order.succ o)
        ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
        inst✝ : Inhabited I
        l : Profinite.NobelingProof.Products I
        hl : Ne (↑l) List.nil
        hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
        hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
        x : ↑(Profinite.NobelingProof.C' C ho)
        hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
        H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
        h₁ : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.NobelingProof.SwapT …
        h₂ : ∀ (i : I), Membership.mem (↑(Profinite.NobelingProof.Products.Tail ⟨List. …
        ⊢ Not (Eq (↑x (Profinite.NobelingProof.term I ho)) Bool.true)
      -/
    · simp [mem_C'_eq_false C ho x x.prop, Bool.coe_false]
      /-
        🎉 no goals
      -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝² : LinearOrder I
      inst✝¹ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      inst✝ : Inhabited I
      l : Profinite.NobelingProof.Products I
      hl : Ne (↑l) List.nil
      hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
      hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
      x : ↑(Profinite.NobelingProof.C' C ho)
      hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
      H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
      h₁ : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.NobelingProof.SwapT …
      h₂ : Not (∀ (i : I), Membership.mem (↑(Profinite.NobelingProof.Products.Tail ⟨ …
      ⊢ Eq (HSub.hSub (ite (Eq (Profinite.NobelingProof.SwapTrue o (↑x) (Profinite.N …
    -/
  · push_neg at h₂; obtain ⟨i, hi⟩ := h₂; exfalso; rw [hi' i hi.1] at hi; exact hi.2 (h₁ i hi.1)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝² : LinearOrder I
      inst✝¹ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      inst✝ : Inhabited I
      l : Profinite.NobelingProof.Products I
      hl : Ne (↑l) List.nil
      hlh : Eq (↑l).head! (Profinite.NobelingProof.term I ho)
      hlc : List.Chain' (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.NobelingPro …
      x : ↑(Profinite.NobelingProof.C' C ho)
      hi' : ∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) (Profinite.NobelingPro …
      H : Eq (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (↑x i) Bool.true) (∀ (i : I …
      h₁ : Not (∀ (i : I), Membership.mem (↑l.Tail) i → Eq (Profinite.NobelingProof. …
      h₃ : ∀ (i : I), Membership.mem (↑(Profinite.NobelingProof.Products.Tail ⟨List. …
      ⊢ Eq 0 1
    -/
  · push_neg at h₁; obtain ⟨i, hi⟩ := h₁; exfalso; rw [← hi' i hi.1] at hi; exact hi.2 (h₃ i hi.1)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem max_eq_eval (l : MaxProducts C ho) :
    Linear_CC' C hsC ho (l.val.eval C) = l.val.Tail.eval (C' C ho) :=
  have : Inhabited I := ⟨term I ho⟩
  Products.max_eq_eval _ _ _ _ (List.ne_nil_of_mem l.prop.2)
    (head!_eq_o_of_maxProducts _ hsC ho l)


theorem max_eq_eval_unapply :
    (Linear_CC' C hsC ho) ∘ (fun (l : MaxProducts C ho) ↦ Products.eval C l.val) =
    (fun l ↦ l.val.Tail.eval (C' C ho)) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ Eq (Function.comp ⇑(Profinite.NobelingProof.Linear_CC' C hsC ho) fun l => Pr …
  -/
  ext1 l
  /-
    case h
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    ⊢ Eq (Function.comp (⇑(Profinite.NobelingProof.Linear_CC' C hsC ho)) (fun l => …
  -/
  exact max_eq_eval _ _ _ _
  /-
    🎉 no goals
  -/


include hsC in
theorem chain'_cons_of_lt (l : MaxProducts C ho)
    (q : Products I) (hq : q < l.val.Tail) :
    List.Chain' (fun x x_1 ↦ x > x_1) (term I ho :: q.val) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    q : Profinite.NobelingProof.Products I
    hq : LT.lt q (↑l).Tail
    ⊢ List.Chain' (fun x x_1 => GT.gt x x_1) (List.cons (Profinite.NobelingProof.t …
  -/
  have : Inhabited I := ⟨term I ho⟩
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    q : Profinite.NobelingProof.Products I
    hq : LT.lt q (↑l).Tail
    this : Inhabited I
    ⊢ List.Chain' (fun x x_1 => GT.gt x x_1) (List.cons (Profinite.NobelingProof.t …
  -/
  rw [List.chain'_iff_pairwise]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    q : Profinite.NobelingProof.Products I
    hq : LT.lt q (↑l).Tail
    this : Inhabited I
    ⊢ List.Pairwise (fun x x_1 => GT.gt x x_1) (List.cons (Profinite.NobelingProof …
  -/
  simp only [gt_iff_lt, List.pairwise_cons]
  refine ⟨fun a ha ↦ lt_of_le_of_lt (Products.rel_head!_of_mem ha) ?_,
    List.chain'_iff_pairwise.mp q.prop⟩
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    q : Profinite.NobelingProof.Products I
    hq : LT.lt q (↑l).Tail
    this : Inhabited I
    a : I
    ha : Membership.mem (↑q) a
    ⊢ LT.lt (↑q).head! (Profinite.NobelingProof.term I ho)
  -/
  refine lt_of_le_of_lt (Products.head!_le_of_lt hq (q.val.ne_nil_of_mem ha)) ?_
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    q : Profinite.NobelingProof.Products I
    hq : LT.lt q (↑l).Tail
    this : Inhabited I
    a : I
    ha : Membership.mem (↑q) a
    ⊢ LT.lt (↑(↑l).Tail).head! (Profinite.NobelingProof.term I ho)
  -/
  by_cases hM : l.val.Tail.val = []
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      q : Profinite.NobelingProof.Products I
      hq : LT.lt q (↑l).Tail
      this : Inhabited I
      a : I
      ha : Membership.mem (↑q) a
      hM : Eq (↑(↑l).Tail) List.nil
      ⊢ LT.lt (↑(↑l).Tail).head! (Profinite.NobelingProof.term I ho)
    -/
  · rw [Products.lt_iff_lex_lt, hM] at hq
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      q : Profinite.NobelingProof.Products I
      hq : List.Lex (fun x1 x2 => LT.lt x1 x2) (↑q) List.nil
      this : Inhabited I
      a : I
      ha : Membership.mem (↑q) a
      hM : Eq (↑(↑l).Tail) List.nil
      ⊢ LT.lt (↑(↑l).Tail).head! (Profinite.NobelingProof.term I ho)
    -/
    simp only [List.Lex.not_nil_right] at hq
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      q : Profinite.NobelingProof.Products I
      hq : LT.lt q (↑l).Tail
      this : Inhabited I
      a : I
      ha : Membership.mem (↑q) a
      hM : Not (Eq (↑(↑l).Tail) List.nil)
      ⊢ LT.lt (↑(↑l).Tail).head! (Profinite.NobelingProof.term I ho)
    -/
  · have := l.val.prop
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      q : Profinite.NobelingProof.Products I
      hq : LT.lt q (↑l).Tail
      this✝ : Inhabited I
      a : I
      ha : Membership.mem (↑q) a
      hM : Not (Eq (↑(↑l).Tail) List.nil)
      this : List.Chain' (fun x1 x2 => GT.gt x1 x2) ↑↑l
      ⊢ LT.lt (↑(↑l).Tail).head! (Profinite.NobelingProof.term I ho)
    -/
    rw [max_eq_o_cons_tail C hsC ho l, List.chain'_iff_pairwise] at this
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      q : Profinite.NobelingProof.Products I
      hq : LT.lt q (↑l).Tail
      this✝ : Inhabited I
      a : I
      ha : Membership.mem (↑q) a
      hM : Not (Eq (↑(↑l).Tail) List.nil)
      this : List.Pairwise (fun x1 x2 => GT.gt x1 x2) (List.cons (Profinite.Nobeling …
      ⊢ LT.lt (↑(↑l).Tail).head! (Profinite.NobelingProof.term I ho)
    -/
    exact List.rel_of_pairwise_cons this (List.head!_mem_self hM)
    /-
      🎉 no goals
    -/


include hsC in
theorem good_lt_maxProducts (q : GoodProducts (π C (ord I · < o)))
    (l : MaxProducts C ho) : List.Lex (·<·) q.val.val l.val.val := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) ↑↑q ↑↑l
  -/
  have : Inhabited I := ⟨term I ho⟩
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    this : Inhabited I
    ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) ↑↑q ↑↑l
  -/
  by_cases h : q.val.val = []
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Eq (↑↑q) List.nil
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) ↑↑q ↑↑l
    -/
  · rw [h, max_eq_o_cons_tail C hsC ho l]
    /-
      case pos
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Eq (↑↑q) List.nil
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) List.nil (List.cons (Profinite.NobelingP …
    -/
    exact List.Lex.nil
    /-
      🎉 no goals
    -/
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Not (Eq (↑↑q) List.nil)
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) ↑↑q ↑↑l
    -/
  · rw [← List.cons_head!_tail h, max_eq_o_cons_tail C hsC ho l]
    /-
      case neg
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Not (Eq (↑↑q) List.nil)
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) (List.cons (↑↑q).head! (↑↑q).tail) (List …
    -/
    apply List.Lex.rel
    /-
      case neg.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Not (Eq (↑↑q) List.nil)
      ⊢ LT.lt (↑↑q).head! (Profinite.NobelingProof.term I ho)
    -/
    rw [← Ordinal.typein_lt_typein (·<·)]
    /-
      case neg.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Not (Eq (↑↑q) List.nil)
      ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding (↑↑q).head!) …
    -/
    simp only [term, Ordinal.typein_enum]
    /-
      case neg.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      q : ↑(Profinite.NobelingProof.GoodProducts (Profinite.NobelingProof.π C fun x  …
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      this : Inhabited I
      h : Not (Eq (↑↑q) List.nil)
      ⊢ LT.lt ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding (↑↑q).head!) o
    -/
    exact Products.prop_of_isGood C _ q.prop q.val.val.head! (List.head!_mem_self h)
    /-
      🎉 no goals
    -/


include hC hsC in
/--
Removing the leading `o` from a term of `MaxProducts C` yields a list which `isGood` with respect to
`C'`.
-/
theorem maxTail_isGood (l : MaxProducts C ho)
    (h₁ : ⊤ ≤ Submodule.span ℤ (Set.range (eval (π C (ord I · < o))))) :
    l.val.Tail.isGood (C' C ho) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    ⊢ Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.C' C ho) (↑ …
  -/
  have : Inhabited I := ⟨term I ho⟩
  -- Write `l.Tail` as a linear combination of smaller products:
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    ⊢ Profinite.NobelingProof.Products.isGood (Profinite.NobelingProof.C' C ho) (↑ …
  -/
  intro h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    h : Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Pro …
    ⊢ False
  -/
  rw [Finsupp.mem_span_image_iff_linearCombination, ← max_eq_eval C hsC ho] at h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    h : Exists fun l_1 => And (Membership.mem (Finsupp.supported Int Int (setOf fu …
    ⊢ False
  -/
  obtain ⟨m, ⟨hmmem, hmsum⟩⟩ := h
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq ((Finsupp.linearCombination Int (Profinite.NobelingProof.Products.e …
    ⊢ False
  -/
  rw [Finsupp.linearCombination_apply] at hmsum

  -- Write the image of `l` under `Linear_CC'` as `Linear_CC'` applied to the linear combination
  -- above, with leading `term I ho`'s added to each term:
  have : (Linear_CC' C hsC ho) (l.val.eval C) = (Linear_CC' C hsC ho)
      (Finsupp.sum m fun i a ↦ a • ((term I ho :: i.1).map (e C)).prod) := by
    rw [← hmsum]
    simp only [map_finsupp_sum]
    apply Finsupp.sum_congr
    intro q hq
    rw [LinearMap.map_smul]
    rw [Finsupp.mem_supported] at hmmem
    have hx'' : q < l.val.Tail := hmmem hq
    have : ∃ (p : Products I), p.val ≠ [] ∧ p.val.head! = term I ho ∧ q = p.Tail :=
      ⟨⟨term I ho :: q.val, chain'_cons_of_lt C hsC ho l q hx''⟩,
        ⟨List.cons_ne_nil _ _, by simp only [List.head!_cons],
        by simp only [Products.Tail, List.tail_cons, Subtype.coe_eta]⟩⟩
    obtain ⟨p, hp⟩ := this
    rw [hp.2.2, ← Products.max_eq_eval C hsC ho p hp.1 hp.2.1]
    dsimp [Products.eval]
    rw [Products.max_eq_o_cons_tail ho p hp.1 hp.2.1]
    rfl
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this✝ : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    this : Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) (Profinite.NobelingPr …
    ⊢ False
  -/
  have hse := succ_exact C hC hsC ho
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this✝ : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    this : Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) (Profinite.NobelingPr …
    hse : (CategoryTheory.ShortComplex.mk (ModuleCat.ofHom (Profinite.NobelingProo …
    ⊢ False
  -/
  rw [ShortComplex.moduleCat_exact_iff_range_eq_ker] at hse
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this✝ : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    this : Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) (Profinite.NobelingPr …
    hse : Eq (LinearMap.range (CategoryTheory.ShortComplex.mk (ModuleCat.ofHom (Pr …
    ⊢ False
  -/
  dsimp [ModuleCat.ofHom] at hse

  -- Rewrite `this` using exact sequence manipulations to conclude that a term is in the range of
  -- the linear map `πs`:
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this✝ : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    this : Eq ((Profinite.NobelingProof.Linear_CC' C hsC ho) (Profinite.NobelingPr …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    ⊢ False
  -/
  rw [← LinearMap.sub_mem_ker_iff, ← hse] at this
  /-
    case intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this✝ : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    this : Membership.mem (LinearMap.range (Profinite.NobelingProof.πs C o)) (HSub …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    ⊢ False
  -/
  obtain ⟨(n : LocallyConstant (π C (ord I · < o)) ℤ), hn⟩ := this
  /-
    case intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hn : Eq ((Profinite.NobelingProof.πs C o) n) (HSub.hSub (Profinite.NobelingPro …
    ⊢ False
  -/
  rw [eq_sub_iff_add_eq] at hn
  /-
    case intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hn : Eq (HAdd.hAdd ((Profinite.NobelingProof.πs C o) n) (m.sum fun i a => HSMu …
    ⊢ False
  -/
  have hn' := h₁ (Submodule.mem_top : n ∈ ⊤)
  /-
    case intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hn : Eq (HAdd.hAdd ((Profinite.NobelingProof.πs C o) n) (m.sum fun i a => HSMu …
    hn' : Membership.mem (Submodule.span Int (Set.range (Profinite.NobelingProof.G …
    ⊢ False
  -/
  rw [Finsupp.mem_span_range_iff_exists_finsupp] at hn'
  /-
    case intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hn : Eq (HAdd.hAdd ((Profinite.NobelingProof.πs C o) n) (m.sum fun i a => HSMu …
    hn' : Exists fun c => Eq (c.sum fun i a => HSMul.hSMul a (Profinite.NobelingPr …
    ⊢ False
  -/
  obtain ⟨w,hc⟩ := hn'
  /-
    case intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    hn : Eq (HAdd.hAdd ((Profinite.NobelingProof.πs C o) n) (m.sum fun i a => HSMu …
    w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
    hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
    ⊢ False
  -/
  rw [← hc, map_finsupp_sum] at hn
  /-
    case intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
    hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
    hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
    ⊢ False
  -/
  apply l.prop.1
  /-
    case intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
    hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
    hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  rw [← hn]

  -- Now we just need to prove that a sum of two terms belongs to a span:
  /-
    case intro.intro.intro.intro
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    this : Inhabited I
    m : Finsupp (Profinite.NobelingProof.Products I) Int
    hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
    hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
    hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
    n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
    w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
    hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
    hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
    ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
  -/
  apply Submodule.add_mem
    /-
      case intro.intro.intro.intro.h₁
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
  · apply Submodule.finsupp_sum_mem
    /-
      case intro.intro.intro.intro.h₁.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      ⊢ ∀ (c : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.N …
    -/
    intro q _
    /-
      case intro.intro.intro.intro.h₁.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    rw [LinearMap.map_smul]
    /-
      case intro.intro.intro.intro.h₁.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    apply Submodule.smul_mem
    /-
      case intro.intro.intro.intro.h₁.h.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    apply Submodule.subset_span
    /-
      case intro.intro.intro.intro.h₁.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (setOf f …
    -/
    dsimp only [eval]
    /-
      case intro.intro.intro.intro.h₁.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (setOf f …
    -/
    rw [Products.eval_πs C (Products.prop_of_isGood _ _ q.prop)]
    /-
      case intro.intro.intro.intro.h₁.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (setOf f …
    -/
    refine ⟨q.val, ⟨?_, rfl⟩⟩
    /-
      case intro.intro.intro.intro.h₁.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ Membership.mem (setOf fun m => LT.lt m ↑l) ↑q
    -/
    simp only [Products.lt_iff_lex_lt, Set.mem_setOf_eq]
    /-
      case intro.intro.intro.intro.h₁.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Subtype fun l => Profinite.NobelingProof.Products.isGood (Profinite.Nobeli …
      a✝ : Ne (w q) 0
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) ↑↑q ↑↑l
    -/
    exact good_lt_maxProducts C hsC ho q l
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.h₂
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
  · apply Submodule.finsupp_sum_mem
    /-
      case intro.intro.intro.intro.h₂.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      ⊢ ∀ (c : Profinite.NobelingProof.Products I), Ne (m c) 0 → Membership.mem (Sub …
    -/
    intro q hq
    /-
      case intro.intro.intro.intro.h₂.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Ne (m q) 0
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    apply Submodule.smul_mem
    /-
      case intro.intro.intro.intro.h₂.h.h
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Ne (m q) 0
      ⊢ Membership.mem (Submodule.span Int (Set.image (Profinite.NobelingProof.Produ …
    -/
    apply Submodule.subset_span
    /-
      case intro.intro.intro.intro.h₂.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : Membership.mem (Finsupp.supported Int Int (setOf fun m => LT.lt m (↑l) …
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Ne (m q) 0
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (setOf f …
    -/
    rw [Finsupp.mem_supported] at hmmem
    /-
      case intro.intro.intro.intro.h₂.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : HasSubset.Subset (↑m.support) (setOf fun m => LT.lt m (↑l).Tail)
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Ne (m q) 0
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (setOf f …
    -/
    rw [← Finsupp.mem_support_iff] at hq
    /-
      case intro.intro.intro.intro.h₂.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : HasSubset.Subset (↑m.support) (setOf fun m => LT.lt m (↑l).Tail)
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Membership.mem m.support q
      ⊢ Membership.mem (Set.image (Profinite.NobelingProof.Products.eval C) (setOf f …
    -/
    refine ⟨⟨term I ho :: q.val, chain'_cons_of_lt C hsC ho l q (hmmem hq)⟩, ⟨?_, rfl⟩⟩
    /-
      case intro.intro.intro.intro.h₂.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : HasSubset.Subset (↑m.support) (setOf fun m => LT.lt m (↑l).Tail)
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Membership.mem m.support q
      ⊢ Membership.mem (setOf fun m => LT.lt m ↑l) ⟨List.cons (Profinite.NobelingPro …
    -/
    simp only [Products.lt_iff_lex_lt, Set.mem_setOf_eq]
    /-
      case intro.intro.intro.intro.h₂.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : HasSubset.Subset (↑m.support) (setOf fun m => LT.lt m (↑l).Tail)
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Membership.mem m.support q
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) (List.cons (Profinite.NobelingProof.term …
    -/
    rw [max_eq_o_cons_tail C hsC ho l]
    /-
      case intro.intro.intro.intro.h₂.h.h.a
      I : Type u
      C : Set (I → Bool)
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      o : Ordinal.{u}
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      l : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
      h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
      this : Inhabited I
      m : Finsupp (Profinite.NobelingProof.Products I) Int
      hmmem : HasSubset.Subset (↑m.support) (setOf fun m => LT.lt m (↑l).Tail)
      hmsum : Eq (m.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.Products.e …
      hse : Eq (LinearMap.range (Profinite.NobelingProof.πs C o)) (LinearMap.ker (Pr …
      n : LocallyConstant (↑(Profinite.NobelingProof.π C fun x => LT.lt (Profinite.N …
      w : Finsupp (Subtype fun l => Profinite.NobelingProof.Products.isGood (Profini …
      hn : Eq (HAdd.hAdd (w.sum fun a b => (Profinite.NobelingProof.πs C o) (HSMul.h …
      hc : Eq (w.sum fun i a => HSMul.hSMul a (Profinite.NobelingProof.GoodProducts. …
      q : Profinite.NobelingProof.Products I
      hq : Membership.mem m.support q
      ⊢ List.Lex (fun x1 x2 => LT.lt x1 x2) (List.cons (Profinite.NobelingProof.term …
    -/
    exact List.Lex.cons ((Products.lt_iff_lex_lt q l.val.Tail).mp (hmmem hq))
    /-
      🎉 no goals
    -/


/-- Given `l : MaxProducts C ho`, its `Tail` is a `GoodProducts (C' C ho)`. -/
noncomputable
def MaxToGood
    (h₁ : ⊤ ≤ Submodule.span ℤ (Set.range (eval (π C (ord I · < o))))) :
    MaxProducts C ho → GoodProducts (C' C ho) :=
  fun l ↦ ⟨l.val.Tail, maxTail_isGood C hC hsC ho l h₁⟩


theorem maxToGood_injective
    (h₁ : ⊤ ≤ Submodule.span ℤ (Set.range (eval (π C (ord I · < o))))) :
    (MaxToGood C hC hsC ho h₁).Injective := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    ⊢ Function.Injective (Profinite.NobelingProof.GoodProducts.MaxToGood C hC hsC  …
  -/
  intro m n h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    m n : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h : Eq (Profinite.NobelingProof.GoodProducts.MaxToGood C hC hsC ho h₁ m) (Prof …
    ⊢ Eq m n
  -/
  apply Subtype.ext ∘ Subtype.ext
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    m n : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h : Eq (Profinite.NobelingProof.GoodProducts.MaxToGood C hC hsC ho h₁ m) (Prof …
    ⊢ Eq ↑↑m ↑↑n
  -/
  rw [Subtype.ext_iff] at h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    m n : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h : Eq ↑(Profinite.NobelingProof.GoodProducts.MaxToGood C hC hsC ho h₁ m) ↑(Pr …
    ⊢ Eq ↑↑m ↑↑n
  -/
  dsimp [MaxToGood] at h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    m n : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho)
    h : Eq (↑m).Tail (↑n).Tail
    ⊢ Eq ↑↑m ↑↑n
  -/
  rw [max_eq_o_cons_tail C hsC ho m, max_eq_o_cons_tail C hsC ho n, h]
  /-
    🎉 no goals
  -/


include hC in
theorem linearIndependent_comp_of_eval
    (h₁ : ⊤ ≤ Submodule.span ℤ (Set.range (eval (π C (ord I · < o))))) :
    LinearIndependent ℤ (eval (C' C ho)) →
    LinearIndependent ℤ (ModuleCat.ofHom (Linear_CC' C hsC ho) ∘ SumEval C ho ∘ Sum.inr) := by
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinite. …
  -/
  dsimp [SumEval, ModuleCat.ofHom]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinite. …
  -/
  rw [max_eq_eval_unapply C hsC ho]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinite. …
  -/
  intro h
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    h : LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinit …
    ⊢ LinearIndependent Int fun l => Profinite.NobelingProof.Products.eval (Profin …
  -/
  let f := MaxToGood C hC hsC ho h₁
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    h : LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinit …
    f : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho) → ↑(Profinite.Nob …
    ⊢ LinearIndependent Int fun l => Profinite.NobelingProof.Products.eval (Profin …
  -/
  have hf : f.Injective := maxToGood_injective C hC hsC ho h₁
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    h : LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinit …
    f : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho) → ↑(Profinite.Nob …
    hf : Function.Injective f
    ⊢ LinearIndependent Int fun l => Profinite.NobelingProof.Products.eval (Profin …
  -/
  have hh : (fun l ↦ Products.eval (C' C ho) l.val.Tail) = eval (C' C ho) ∘ f := rfl
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    h : LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinit …
    f : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho) → ↑(Profinite.Nob …
    hf : Function.Injective f
    hh : Eq (fun l => Profinite.NobelingProof.Products.eval (Profinite.NobelingPro …
    ⊢ LinearIndependent Int fun l => Profinite.NobelingProof.Products.eval (Profin …
  -/
  rw [hh]
  /-
    I : Type u
    C : Set (I → Bool)
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    h₁ : LE.le Top.top (Submodule.span Int (Set.range (Profinite.NobelingProof.Goo …
    h : LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinit …
    f : ↑(Profinite.NobelingProof.GoodProducts.MaxProducts C ho) → ↑(Profinite.Nob …
    hf : Function.Injective f
    hh : Eq (fun l => Profinite.NobelingProof.Products.eval (Profinite.NobelingPro …
    ⊢ LinearIndependent Int (Function.comp (Profinite.NobelingProof.GoodProducts.e …
  -/
  exact h.comp f hf
  /-
    🎉 no goals
  -/


theorem GoodProducts.P0 : P I 0 := fun _ C _ hsC ↦ by
  have : C ⊆ {(fun _ ↦ false)} := fun c hc ↦ by
    ext x; exact Bool.eq_false_iff.mpr (fun ht ↦ (Ordinal.not_lt_zero (ord I x)) (hsC c hc x ht))
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    x✝¹ : LE.le 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    C : Set (I → Bool)
    x✝ : IsClosed C
    hsC : Profinite.NobelingProof.contained C 0
    this : HasSubset.Subset C (Singleton.singleton fun x => Bool.false)
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)
  -/
  rw [Set.subset_singleton_iff_eq] at this
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    x✝¹ : LE.le 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    C : Set (I → Bool)
    x✝ : IsClosed C
    hsC : Profinite.NobelingProof.contained C 0
    this : Or (Eq C EmptyCollection.emptyCollection) (Eq C (Singleton.singleton fu …
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)
  -/
  cases this
    /-
      case inl
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      x✝¹ : LE.le 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      C : Set (I → Bool)
      x✝ : IsClosed C
      hsC : Profinite.NobelingProof.contained C 0
      h✝ : Eq C EmptyCollection.emptyCollection
      ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)
    -/
  · subst C
    /-
      case inl
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      x✝¹ : LE.le 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x✝ : IsClosed EmptyCollection.emptyCollection
      hsC : Profinite.NobelingProof.contained EmptyCollection.emptyCollection 0
      ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval EmptyCollec …
    -/
    exact linearIndependentEmpty
    /-
      🎉 no goals
    -/
    /-
      case inr
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      x✝¹ : LE.le 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      C : Set (I → Bool)
      x✝ : IsClosed C
      hsC : Profinite.NobelingProof.contained C 0
      h✝ : Eq C (Singleton.singleton fun x => Bool.false)
      ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)
    -/
  · subst C
    /-
      case inr
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      x✝¹ : LE.le 0 (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      x✝ : IsClosed (Singleton.singleton fun x => Bool.false)
      hsC : Profinite.NobelingProof.contained (Singleton.singleton fun x => Bool.fal …
      ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Singleton. …
    -/
    exact linearIndependentSingleton
    /-
      🎉 no goals
    -/


theorem GoodProducts.Plimit (o : Ordinal) (ho : Ordinal.IsLimit o) :
    (∀ (o' : Ordinal), o' < o → P I o') → P I o := by
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    ⊢ (∀ (o' : Ordinal.{u}), LT.lt o' o → Profinite.NobelingProof.P I o') → Profin …
  -/
  intro h hho C hC hsC
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    o : Ordinal.{u}
    ho : o.IsLimit
    h : ∀ (o' : Ordinal.{u}), LT.lt o' o → Profinite.NobelingProof.P I o'
    hho : LE.le o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    C : Set (I → Bool)
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C o
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)
  -/
  rw [linearIndependent_iff_union_smaller C ho hsC]
  exact linearIndependent_iUnion_of_directed
    (Monotone.directed_le fun _ _ h ↦ GoodProducts.smaller_mono C h) fun ⟨o', ho'⟩ ↦
    (linearIndependent_iff_smaller _ _).mp (h o' ho' (le_of_lt (lt_of_lt_of_le ho' hho))
    (π C (ord I · < o')) (isClosed_proj _ _ hC) (contained_proj _ _))


theorem GoodProducts.linearIndependentAux (μ : Ordinal) : P I μ := by
  refine Ordinal.limitRecOn μ P0 (fun o h ho C hC hsC ↦ ?_)
      (fun o ho h ↦ (GoodProducts.Plimit o ho (fun o' ho' ↦ (h o' ho'))))
  have ho' : o < Ordinal.type (·<· : I → I → Prop) :=
    lt_of_lt_of_le (Order.lt_succ _) ho
  /-
    I : Type u
    inst✝¹ : LinearOrder I
    inst✝ : WellFoundedLT I
    μ o : Ordinal.{u}
    h : Profinite.NobelingProof.P I o
    ho : LE.le (Order.succ o) (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    C : Set (I → Bool)
    hC : IsClosed C
    hsC : Profinite.NobelingProof.contained C (Order.succ o)
    ho' : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
    ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval C)
  -/
  rw [linearIndependent_iff_sum C hsC ho']
  refine ModuleCat.linearIndependent_leftExact (succ_exact C hC hsC ho') ?_ ?_ (succ_mono C o)
    (square_commutes C ho')
    /-
      case refine_1
      I : Type u
      inst✝¹ : LinearOrder I
      inst✝ : WellFoundedLT I
      μ o : Ordinal.{u}
      h : Profinite.NobelingProof.P I o
      ho : LE.le (Order.succ o) (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      C : Set (I → Bool)
      hC : IsClosed C
      hsC : Profinite.NobelingProof.contained C (Order.succ o)
      ho' : LT.lt o (Ordinal.type fun x1 x2 => LT.lt x1 x2)
      ⊢ LinearIndependent Int (Profinite.NobelingProof.GoodProducts.eval (Profinite. …
    -/
  · exact h (le_of_lt ho') (π C (ord I · < o)) (isClosed_proj C o hC) (contained_proj C o)
    /-
      🎉 no goals
    -/
  · exact linearIndependent_comp_of_eval C hC hsC ho' (span (π C (ord I · < o))
      (isClosed_proj C o hC)) (h (le_of_lt ho') (C' C ho') (isClosed_C' C hC ho')
      (contained_C' C ho'))


theorem GoodProducts.linearIndependent (hC : IsClosed C) :
    LinearIndependent ℤ (GoodProducts.eval C) :=
  GoodProducts.linearIndependentAux (Ordinal.type (·<· : I → I → Prop)) (le_refl _)
    C hC (fun _ _ _ _ ↦ Ordinal.typein_lt_type _ _)


/-- `GoodProducts C` as a `ℤ`-basis for `LocallyConstant C ℤ`. -/
noncomputable
def GoodProducts.Basis (hC : IsClosed C) :
    Basis (GoodProducts C) ℤ (LocallyConstant C ℤ) :=
  Basis.mk (GoodProducts.linearIndependent C hC) (GoodProducts.span C hC)


/--
Given a profinite set `S` and a closed embedding `S → (I → Bool)`, the `ℤ`-module
`LocallyConstant C ℤ` is free.
-/
theorem Nobeling_aux : Module.Free ℤ (LocallyConstant S ℤ) := Module.Free.of_equiv'
  (Module.Free.of_basis <| GoodProducts.Basis _ hι.isClosed_range) (LocallyConstant.congrLeftₗ ℤ
    (.ofIsEmbedding ι hι.isEmbedding)).symm


open scoped Classical in
/-- The embedding `S → (I → Bool)` where `I` is the set of clopens of `S`. -/
noncomputable
def Nobeling.ι : S → ({C : Set S // IsClopen C} → Bool) := fun s C => decide (s ∈ C.1)


open scoped Classical in
/-- The map `Nobeling.ι` is a closed embedding. -/
theorem Nobeling.isClosedEmbedding : IsClosedEmbedding (Nobeling.ι S) := by
  /-
    S : Profinite
    ⊢ Topology.IsClosedEmbedding (Profinite.Nobeling.ι S)
  -/
  apply Continuous.isClosedEmbedding
    /-
      case h
      S : Profinite
      ⊢ Continuous (Profinite.Nobeling.ι S)
    -/
  · dsimp (config := { unfoldPartialApp := true }) [ι]
    /-
      case h
      S : Profinite
      ⊢ Continuous fun s C => Decidable.decide (Membership.mem (↑C) s)
    -/
    refine continuous_pi ?_
    /-
      case h
      S : Profinite
      ⊢ ∀ (i : Subtype fun C => IsClopen C), Continuous fun a => Decidable.decide (M …
    -/
    intro C
    /-
      case h
      S : Profinite
      C : Subtype fun C => IsClopen C
      ⊢ Continuous fun a => Decidable.decide (Membership.mem (↑C) a)
    -/
    rw [← IsLocallyConstant.iff_continuous]
    /-
      case h
      S : Profinite
      C : Subtype fun C => IsClopen C
      ⊢ IsLocallyConstant fun a => Decidable.decide (Membership.mem (↑C) a)
    -/
    refine ((IsLocallyConstant.tfae _).out 0 3).mpr ?_
    /-
      case h
      S : Profinite
      C : Subtype fun C => IsClopen C
      ⊢ ∀ (y : Bool), IsOpen (Set.preimage (fun a => Decidable.decide (Membership.me …
    -/
    rintro ⟨⟩
      /-
        case h.false
        S : Profinite
        C : Subtype fun C => IsClopen C
        ⊢ IsOpen (Set.preimage (fun a => Decidable.decide (Membership.mem (↑C) a)) (Si …
      -/
    · refine IsClopen.isOpen (isClopen_compl_iff.mp ?_)
      /-
        case h.false
        S : Profinite
        C : Subtype fun C => IsClopen C
        ⊢ IsClopen (HasCompl.compl (Set.preimage (fun a => Decidable.decide (Membershi …
      -/
      convert C.2
      /-
        case h.e'_3
        S : Profinite
        C : Subtype fun C => IsClopen C
        ⊢ Eq (HasCompl.compl (Set.preimage (fun a => Decidable.decide (Membership.mem  …
      -/
      ext x
      simp only [Set.mem_compl_iff, Set.mem_preimage, Set.mem_singleton_iff,
        decide_eq_false_iff_not, not_not]
      /-
        case h.true
        S : Profinite
        C : Subtype fun C => IsClopen C
        ⊢ IsOpen (Set.preimage (fun a => Decidable.decide (Membership.mem (↑C) a)) (Si …
      -/
    · refine IsClopen.isOpen ?_
      /-
        case h.true
        S : Profinite
        C : Subtype fun C => IsClopen C
        ⊢ IsClopen (Set.preimage (fun a => Decidable.decide (Membership.mem (↑C) a)) ( …
      -/
      convert C.2
      /-
        case h.e'_3
        S : Profinite
        C : Subtype fun C => IsClopen C
        ⊢ Eq (Set.preimage (fun a => Decidable.decide (Membership.mem (↑C) a)) (Single …
      -/
      ext x
      /-
        case h.e'_3.h
        S : Profinite
        C : Subtype fun C => IsClopen C
        x : ↑S.toTop
        ⊢ Iff (Membership.mem (Set.preimage (fun a => Decidable.decide (Membership.mem …
      -/
      simp only [Set.mem_preimage, Set.mem_singleton_iff, decide_eq_true_eq]
      /-
        🎉 no goals
      -/
    /-
      case hf
      S : Profinite
      ⊢ Function.Injective (Profinite.Nobeling.ι S)
    -/
  · intro a b h
    /-
      case hf
      S : Profinite
      a b : ↑S.toTop
      h : Eq (Profinite.Nobeling.ι S a) (Profinite.Nobeling.ι S b)
      ⊢ Eq a b
    -/
    by_contra hn
    /-
      case hf
      S : Profinite
      a b : ↑S.toTop
      h : Eq (Profinite.Nobeling.ι S a) (Profinite.Nobeling.ι S b)
      hn : Not (Eq a b)
      ⊢ False
    -/
    obtain ⟨C, hC, hh⟩ := exists_isClopen_of_totally_separated hn
    /-
      case hf.intro.intro
      S : Profinite
      a b : ↑S.toTop
      h : Eq (Profinite.Nobeling.ι S a) (Profinite.Nobeling.ι S b)
      hn : Not (Eq a b)
      C : Set ↑S.toTop
      hC : IsClopen C
      hh : And (Membership.mem C a) (Membership.mem (HasCompl.compl C) b)
      ⊢ False
    -/
    apply hh.2 ∘ of_decide_eq_true
    /-
      case hf.intro.intro
      S : Profinite
      a b : ↑S.toTop
      h : Eq (Profinite.Nobeling.ι S a) (Profinite.Nobeling.ι S b)
      hn : Not (Eq a b)
      C : Set ↑S.toTop
      hC : IsClopen C
      hh : And (Membership.mem C a) (Membership.mem (HasCompl.compl C) b)
      ⊢ Eq (Decidable.decide (Membership.mem C b)) Bool.true
    -/
    dsimp (config := { unfoldPartialApp := true }) [ι] at h
    /-
      case hf.intro.intro
      S : Profinite
      a b : ↑S.toTop
      h : Eq (fun C => Decidable.decide (Membership.mem (↑C) a)) fun C => Decidable. …
      hn : Not (Eq a b)
      C : Set ↑S.toTop
      hC : IsClopen C
      hh : And (Membership.mem C a) (Membership.mem (HasCompl.compl C) b)
      ⊢ Eq (Decidable.decide (Membership.mem C b)) Bool.true
    -/
    rw [← congr_fun h ⟨C, hC⟩]
    /-
      case hf.intro.intro
      S : Profinite
      a b : ↑S.toTop
      h : Eq (fun C => Decidable.decide (Membership.mem (↑C) a)) fun C => Decidable. …
      hn : Not (Eq a b)
      C : Set ↑S.toTop
      hC : IsClopen C
      hh : And (Membership.mem C a) (Membership.mem (HasCompl.compl C) b)
      ⊢ Eq (Decidable.decide (Membership.mem (↑⟨C, hC⟩) a)) Bool.true
    -/
    exact decide_eq_true hh.1
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias Nobeling.embedding := Nobeling.isClosedEmbedding


/-- Nöbeling's theorem: the `ℤ`-module `LocallyConstant S ℤ` is free for every `S : Profinite` -/
instance LocallyConstant.freeOfProfinite (S : Profinite.{u}) :
    Module.Free ℤ (LocallyConstant S ℤ) := by
  /-
    S : Profinite
    ⊢ Module.Free Int (LocallyConstant (↑S.toTop) Int)
  -/
  obtain ⟨_, _⟩ := exists_wellOrder {C : Set S // IsClopen C}
  /-
    case intro
    S : Profinite
    w✝ : LinearOrder (Subtype fun C => IsClopen C)
    h✝ : WellFoundedLT (Subtype fun C => IsClopen C)
    ⊢ Module.Free Int (LocallyConstant (↑S.toTop) Int)
  -/
  exact @Nobeling_aux {C : Set S // IsClopen C} _ _ S (Nobeling.ι S) (Nobeling.isClosedEmbedding S)
  /-
    🎉 no goals
  -/


