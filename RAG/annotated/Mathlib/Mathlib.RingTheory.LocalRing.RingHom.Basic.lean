@[instance]
theorem isLocalHom_id (R : Type*) [Semiring R] : IsLocalHom (RingHom.id R) where
  map_nonunit _ := id


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_id := isLocalHom_id

-- see note [lower instance priority]

@[instance 100]
theorem isLocalHom_toRingHom {F : Type*} [FunLike F R S]
    [RingHomClass F R S] (f : F) [IsLocalHom f] : IsLocalHom (f : R →+* S) :=
  ⟨IsLocalHom.map_nonunit (f := f)⟩


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_toRingHom := isLocalHom_toRingHom


@[instance]
theorem RingHom.isLocalHom_comp (g : S →+* T) (f : R →+* S) [IsLocalHom g]
    [IsLocalHom f] : IsLocalHom (g.comp f) where
  map_nonunit a := IsLocalHom.map_nonunit a ∘ IsLocalHom.map_nonunit (f := g) (f a)


@[deprecated (since := "2024-10-10")]
alias RingHom.isLocalRingHom_comp := RingHom.isLocalHom_comp


theorem isLocalHom_of_comp (f : R →+* S) (g : S →+* T) [IsLocalHom (g.comp f)] :
    IsLocalHom f :=
  ⟨fun _ ha => (isUnit_map_iff (g.comp f) _).mp (g.isUnit_map ha)⟩


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_of_comp := isLocalHom_of_comp


/-- If `f : R →+* S` is a local ring hom, then `R` is a local ring if `S` is. -/
theorem RingHom.domain_isLocalRing {R S : Type*} [CommSemiring R] [CommSemiring S] [IsLocalRing S]
    (f : R →+* S) [IsLocalHom f] : IsLocalRing R := by
  /-
    R : Type u_4
    S : Type u_5
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : IsLocalRing S
    f : RingHom R S
    inst✝ : IsLocalHom f
    ⊢ IsLocalRing R
  -/
  haveI : Nontrivial R := f.domain_nontrivial
  /-
    R : Type u_4
    S : Type u_5
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : IsLocalRing S
    f : RingHom R S
    inst✝ : IsLocalHom f
    this : Nontrivial R
    ⊢ IsLocalRing R
  -/
  apply IsLocalRing.of_nonunits_add
  /-
    case h
    R : Type u_4
    S : Type u_5
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : IsLocalRing S
    f : RingHom R S
    inst✝ : IsLocalHom f
    this : Nontrivial R
    ⊢ ∀ (a b : R), Membership.mem (nonunits R) a → Membership.mem (nonunits R) b → …
  -/
  intro a b
  /-
    case h
    R : Type u_4
    S : Type u_5
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : IsLocalRing S
    f : RingHom R S
    inst✝ : IsLocalHom f
    this : Nontrivial R
    a b : R
    ⊢ Membership.mem (nonunits R) a → Membership.mem (nonunits R) b → Membership.m …
  -/
  simp_rw [← map_mem_nonunits_iff f, f.map_add]
  /-
    case h
    R : Type u_4
    S : Type u_5
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : IsLocalRing S
    f : RingHom R S
    inst✝ : IsLocalHom f
    this : Nontrivial R
    a b : R
    ⊢ Membership.mem (nonunits S) (f a) → Membership.mem (nonunits S) (f b) → Memb …
  -/
  exact IsLocalRing.nonunits_add
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-12")] alias RingHom.domain_localRing := RingHom.domain_isLocalRing


/--
The image of the maximal ideal of the source is contained within the maximal ideal of the target.
-/
theorem map_nonunit (f : R →+* S) [IsLocalHom f] (a : R) (h : a ∈ maximalIdeal R) :
    f a ∈ maximalIdeal S := fun H => h <| isUnit_of_map_unit f a H


/-- A ring homomorphism between local rings is a local ring hom iff it reflects units,
i.e. any preimage of a unit is still a unit. https://stacks.math.columbia.edu/tag/07BJ
-/
theorem local_hom_TFAE (f : R →+* S) :
    List.TFAE
      [IsLocalHom f, f '' (maximalIdeal R).1 ⊆ maximalIdeal S,
        (maximalIdeal R).map f ≤ maximalIdeal S, maximalIdeal R ≤ (maximalIdeal S).comap f,
        (maximalIdeal S).comap f = maximalIdeal R] := by
  tfae_have 1 → 2
  | _, _, ⟨a, ha, rfl⟩ => map_nonunit f a ha
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    tfae_1_to_2 : IsLocalHom f → HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maxi …
    ⊢ (List.cons (IsLocalHom f) (List.cons (HasSubset.Subset (Set.image ⇑f ↑(IsLoc …
  -/
  tfae_have 2 → 4 := Set.image_subset_iff.1
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    tfae_1_to_2 : IsLocalHom f → HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maxi …
    tfae_2_to_4 : HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maximalIdeal R).toA …
    ⊢ (List.cons (IsLocalHom f) (List.cons (HasSubset.Subset (Set.image ⇑f ↑(IsLoc …
  -/
  tfae_have 3 ↔ 4 := Ideal.map_le_iff_le_comap
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    tfae_1_to_2 : IsLocalHom f → HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maxi …
    tfae_2_to_4 : HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maximalIdeal R).toA …
    tfae_3_iff_4 : Iff (LE.le (Ideal.map f (IsLocalRing.maximalIdeal R)) (IsLocalR …
    ⊢ (List.cons (IsLocalHom f) (List.cons (HasSubset.Subset (Set.image ⇑f ↑(IsLoc …
  -/
  tfae_have 4 → 1 := fun h ↦ ⟨fun x => not_imp_not.1 (@h x)⟩
  tfae_have 1 → 5
  | _ => by ext; exact not_iff_not.2 (isUnit_map_iff f _)
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    tfae_1_to_2 : IsLocalHom f → HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maxi …
    tfae_2_to_4 : HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maximalIdeal R).toA …
    tfae_3_iff_4 : Iff (LE.le (Ideal.map f (IsLocalRing.maximalIdeal R)) (IsLocalR …
    tfae_4_to_1 : LE.le (IsLocalRing.maximalIdeal R) (Ideal.comap f (IsLocalRing.m …
    tfae_1_to_5 : IsLocalHom f → Eq (Ideal.comap f (IsLocalRing.maximalIdeal S)) ( …
    ⊢ (List.cons (IsLocalHom f) (List.cons (HasSubset.Subset (Set.image ⇑f ↑(IsLoc …
  -/
  tfae_have 5 → 4 := fun h ↦ le_of_eq h.symm
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsLocalRing R
    inst✝¹ : CommSemiring S
    inst✝ : IsLocalRing S
    f : RingHom R S
    tfae_1_to_2 : IsLocalHom f → HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maxi …
    tfae_2_to_4 : HasSubset.Subset (Set.image ⇑f ↑(IsLocalRing.maximalIdeal R).toA …
    tfae_3_iff_4 : Iff (LE.le (Ideal.map f (IsLocalRing.maximalIdeal R)) (IsLocalR …
    tfae_4_to_1 : LE.le (IsLocalRing.maximalIdeal R) (Ideal.comap f (IsLocalRing.m …
    tfae_1_to_5 : IsLocalHom f → Eq (Ideal.comap f (IsLocalRing.maximalIdeal S)) ( …
    tfae_5_to_4 : Eq (Ideal.comap f (IsLocalRing.maximalIdeal S)) (IsLocalRing.max …
    ⊢ (List.cons (IsLocalHom f) (List.cons (HasSubset.Subset (Set.image ⇑f ↑(IsLoc …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem of_surjective [CommSemiring R] [IsLocalRing R] [CommSemiring S] [Nontrivial S] (f : R →+* S)
    [IsLocalHom f] (hf : Function.Surjective f) : IsLocalRing S :=
  of_isUnit_or_isUnit_of_isUnit_add (by
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : IsLocalRing R
      inst✝² : CommSemiring S
      inst✝¹ : Nontrivial S
      f : RingHom R S
      inst✝ : IsLocalHom f
      hf : Function.Surjective ⇑f
      ⊢ ∀ (a b : S), IsUnit (HAdd.hAdd a b) → Or (IsUnit a) (IsUnit b)
    -/
    intro a b hab
    /-
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : IsLocalRing R
      inst✝² : CommSemiring S
      inst✝¹ : Nontrivial S
      f : RingHom R S
      inst✝ : IsLocalHom f
      hf : Function.Surjective ⇑f
      a b : S
      hab : IsUnit (HAdd.hAdd a b)
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
    obtain ⟨a, rfl⟩ := hf a
    /-
      case intro
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : IsLocalRing R
      inst✝² : CommSemiring S
      inst✝¹ : Nontrivial S
      f : RingHom R S
      inst✝ : IsLocalHom f
      hf : Function.Surjective ⇑f
      b : S
      a : R
      hab : IsUnit (HAdd.hAdd (f a) b)
      ⊢ Or (IsUnit (f a)) (IsUnit b)
    -/
    obtain ⟨b, rfl⟩ := hf b
    /-
      case intro.intro
      R : Type u_1
      S : Type u_2
      inst✝⁴ : CommSemiring R
      inst✝³ : IsLocalRing R
      inst✝² : CommSemiring S
      inst✝¹ : Nontrivial S
      f : RingHom R S
      inst✝ : IsLocalHom f
      hf : Function.Surjective ⇑f
      a b : R
      hab : IsUnit (HAdd.hAdd (f a) (f b))
      ⊢ Or (IsUnit (f a)) (IsUnit (f b))
    -/
    rw [← map_add] at hab
    exact
      (isUnit_or_isUnit_of_isUnit_add <| IsLocalHom.map_nonunit _ hab).imp f.isUnit_map
        f.isUnit_map)


lemma _root_.IsLocalHom.of_surjective [CommRing R] [CommRing S] [Nontrivial S] [IsLocalRing R]
    (f : R →+* S) (hf : Function.Surjective f) :
    IsLocalHom f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Nontrivial S
    inst✝ : IsLocalRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    ⊢ IsLocalHom f
  -/
  have := IsLocalRing.of_surjective' f ‹_›
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Nontrivial S
    inst✝ : IsLocalRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    this : IsLocalRing S
    ⊢ IsLocalHom f
  -/
  refine ((local_hom_TFAE f).out 3 0).mp ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Nontrivial S
    inst✝ : IsLocalRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    this : IsLocalRing S
    ⊢ LE.le (IsLocalRing.maximalIdeal R) (Ideal.comap f (IsLocalRing.maximalIdeal  …
  -/
  have := Ideal.comap_isMaximal_of_surjective f hf (K := maximalIdeal S)
  /-
    R : Type u_1
    S : Type u_2
    inst✝³ : CommRing R
    inst✝² : CommRing S
    inst✝¹ : Nontrivial S
    inst✝ : IsLocalRing R
    f : RingHom R S
    hf : Function.Surjective ⇑f
    this✝ : IsLocalRing S
    this : (Ideal.comap f (IsLocalRing.maximalIdeal S)).IsMaximal
    ⊢ LE.le (IsLocalRing.maximalIdeal R) (Ideal.comap f (IsLocalRing.maximalIdeal  …
  -/
  exact ((maximal_ideal_unique R).unique (inferInstanceAs (maximalIdeal R).IsMaximal) this).le
  /-
    🎉 no goals
  -/


alias _root_.Function.Surjective.isLocalHom := _root_.IsLocalHom.of_surjective


/-- If `f : R →+* S` is a surjective local ring hom, then the induced units map is surjective. -/
theorem surjective_units_map_of_local_ringHom [CommRing R] [CommRing S] (f : R →+* S)
    (hf : Function.Surjective f) (h : IsLocalHom f) :
    Function.Surjective (Units.map <| f.toMonoidHom) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : IsLocalHom f
    ⊢ Function.Surjective ⇑(Units.map ↑f)
  -/
  intro a
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : IsLocalHom f
    a : Units S
    ⊢ Exists fun a_1 => Eq ((Units.map ↑f) a_1) a
  -/
  obtain ⟨b, hb⟩ := hf (a : S)
  /-
    case intro
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : IsLocalHom f
    a : Units S
    b : R
    hb : Eq (f b) ↑a
    ⊢ Exists fun a_1 => Eq ((Units.map ↑f) a_1) a
  -/
  use (isUnit_of_map_unit f b (by rw [hb]; exact Units.isUnit _)).unit
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : IsLocalHom f
    a : Units S
    b : R
    hb : Eq (f b) ↑a
    ⊢ Eq ((Units.map ↑f) ⋯.unit) a
  -/
  ext
  /-
    case h.a
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    hf : Function.Surjective ⇑f
    h : IsLocalHom f
    a : Units S
    b : R
    hb : Eq (f b) ↑a
    ⊢ Eq ↑((Units.map ↑f) ⋯.unit) ↑a
  -/
  exact hb
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

/-- Every ring hom `f : K →+* R` from a division ring `K` to a nontrivial ring `R` is a
local ring hom. -/
instance (priority := 100) {K R} [DivisionRing K] [CommRing R] [Nontrivial R]
    (f : K →+* R) : IsLocalHom f where
                         /-
                           R✝ : Type u_1
                           S : Type u_2
                           T : Type u_3
                           K : Type u_4
                           R : Type u_5
                           inst✝² : DivisionRing K
                           inst✝¹ : CommRing R
                           inst✝ : Nontrivial R
                           f : RingHom K R
                           r : K
                           hr : IsUnit (f r)
                           ⊢ IsUnit r
                         -/
  map_nonunit r hr := by simpa only [isUnit_iff_ne_zero, ne_eq, map_eq_zero] using hr.ne_zero
                         /-
                           🎉 no goals
                         -/


@[deprecated (since := "2024-11-11")] alias LocalRing.local_hom_TFAE := IsLocalRing.local_hom_TFAE

@[deprecated (since := "2024-11-11")] alias LocalRing.of_surjective := IsLocalRing.of_surjective

@[deprecated (since := "2024-11-11")]
alias LocalRing.surjective_units_map_of_local_ringHom :=
  IsLocalRing.surjective_units_map_of_local_ringHom


protected theorem isLocalRing {A B : Type*} [CommSemiring A] [IsLocalRing A] [CommSemiring B]
    (e : A ≃+* B) : IsLocalRing B :=
  haveI := e.symm.toEquiv.nontrivial
  IsLocalRing.of_surjective (e : A →+* B) e.surjective


@[deprecated (since := "2024-11-09")] alias localRing := RingEquiv.isLocalRing


