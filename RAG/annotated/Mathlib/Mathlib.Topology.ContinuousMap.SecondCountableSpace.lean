theorem compactOpen_eq_generateFrom {S : Set (Set X)} {T : Set (Set Y)}
    (hS₁ : ∀ K ∈ S, IsCompact K) (hT : IsTopologicalBasis T)
    (hS₂ : ∀ f : C(X, Y), ∀ x, ∀ V ∈ T, f x ∈ V → ∃ K ∈ S, K ∈ 𝓝 x ∧ MapsTo f K V) :
    compactOpen = .generateFrom (.image2 (fun K t ↦
      {f : C(X, Y) | MapsTo f K (⋃₀ t)}) S {t : Set (Set Y) | t.Finite ∧ t ⊆ T}) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    S : Set (Set X)
    T : Set (Set Y)
    hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
    hT : TopologicalSpace.IsTopologicalBasis T
    hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
    ⊢ Eq ContinuousMap.compactOpen (TopologicalSpace.generateFrom (Set.image2 (fun …
  -/
  apply le_antisymm
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      ⊢ LE.le ContinuousMap.compactOpen (TopologicalSpace.generateFrom (Set.image2 ( …
    -/
  · apply_rules [generateFrom_anti, image2_subset_iff.mpr]
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      ⊢ ∀ (x : Set X), Membership.mem S x → ∀ (y : Set (Set Y)), Membership.mem (set …
    -/
    intro K hK t ht
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      K : Set X
      hK : Membership.mem S K
      t : Set (Set Y)
      ht : Membership.mem (setOf fun t => And t.Finite (HasSubset.Subset t T)) t
      ⊢ Membership.mem (Set.image2 (fun K U => setOf fun f => Set.MapsTo (⇑f) K U) ( …
    -/
    exact mem_image2_of_mem (hS₁ K hK) (isOpen_sUnion fun _ h ↦ hT.isOpen <| ht.2 h)
    /-
      🎉 no goals
    -/
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      ⊢ LE.le (TopologicalSpace.generateFrom (Set.image2 (fun K t => setOf fun f =>  …
    -/
  · refine le_of_nhds_le_nhds fun f ↦ ?_
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      ⊢ LE.le (nhds f) (nhds f)
    -/
    simp only [nhds_compactOpen, le_iInf_iff, le_principal_iff]
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      ⊢ ∀ (i : Set X), IsCompact i → ∀ (i_2 : Set Y), IsOpen i_2 → Set.MapsTo (⇑f) i …
    -/
    intro K (hK : IsCompact K) U (hU : IsOpen U) hfKU
    /-
      case a
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hfKU : Set.MapsTo (⇑f) K U
      ⊢ Membership.mem (nhds f) (setOf fun g => Set.MapsTo (⇑g) K U)
    -/
    simp only [TopologicalSpace.nhds_generateFrom]
    obtain ⟨t, htT, htf, hTU, hKT⟩ : ∃ t ⊆ T, t.Finite ∧ (∀ V ∈ t, V ⊆ U) ∧ f '' K ⊆ ⋃₀ t := by
      rw [hT.open_eq_sUnion' hU, mapsTo', sUnion_eq_biUnion] at hfKU
      obtain ⟨t, ht, hfin, htK⟩ :=
        (hK.image (map_continuous f)).elim_finite_subcover_image (fun V hV ↦ hT.isOpen hV.1) hfKU
      refine ⟨t, fun _ h ↦ (ht h).1, hfin, fun _ h ↦ (ht h).2, ?_⟩
      rwa [sUnion_eq_biUnion]
    /-
      case a.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hfKU : Set.MapsTo (⇑f) K U
      t : Set (Set Y)
      htT : HasSubset.Subset t T
      htf : t.Finite
      hTU : ∀ (V : Set Y), Membership.mem t V → HasSubset.Subset V U
      hKT : HasSubset.Subset (Set.image (⇑f) K) t.sUnion
      ⊢ Membership.mem (iInf fun s => iInf fun h => Filter.principal s) (setOf fun g …
    -/
    rw [image_subset_iff] at hKT
    obtain ⟨s, hsS, hsf, hKs, hst⟩ : ∃ s ⊆ S, s.Finite ∧ K ⊆ ⋃₀ s ∧ MapsTo f (⋃₀ s) (⋃₀ t) := by
      have : ∀ x ∈ K, ∃ L ∈ S, L ∈ 𝓝 x ∧ MapsTo f L (⋃₀ t) := by
        intro x hx
        rcases hKT hx with ⟨V, hVt, hxV⟩
        rcases hS₂ f x V (htT hVt) hxV with ⟨L, hLS, hLx, hLV⟩
        exact ⟨L, hLS, hLx, hLV.mono_right <| subset_sUnion_of_mem hVt⟩
      choose! L hLS hLmem hLt using this
      rcases hK.elim_nhds_subcover L hLmem with ⟨s, hsK, hs⟩
      refine ⟨L '' s, image_subset_iff.2 fun x hx ↦ hLS x <| hsK x hx, s.finite_toSet.image _,
        by rwa [sUnion_image], ?_⟩
      rw [mapsTo_sUnion, forall_mem_image]
      exact fun x hx ↦ hLt x <| hsK x hx
    have hsub : (⋂ L ∈ s, {g : C(X, Y) | MapsTo g L (⋃₀ t)}) ⊆ {g | MapsTo g K U} := by
      simp only [← setOf_forall, ← mapsTo_iUnion, ← sUnion_eq_biUnion]
      exact fun g hg ↦ hg.mono hKs (sUnion_subset hTU)
    /-
      case a.intro.intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hfKU : Set.MapsTo (⇑f) K U
      t : Set (Set Y)
      htT : HasSubset.Subset t T
      htf : t.Finite
      hTU : ∀ (V : Set Y), Membership.mem t V → HasSubset.Subset V U
      hKT : HasSubset.Subset K (Set.preimage (⇑f) t.sUnion)
      s : Set (Set X)
      hsS : HasSubset.Subset s S
      hsf : s.Finite
      hKs : HasSubset.Subset K s.sUnion
      hst : Set.MapsTo (⇑f) s.sUnion t.sUnion
      hsub : HasSubset.Subset (Set.iInter fun L => Set.iInter fun h => setOf fun g = …
      ⊢ Membership.mem (iInf fun s => iInf fun h => Filter.principal s) (setOf fun g …
    -/
    refine mem_of_superset ((biInter_mem hsf).2 fun L hL ↦ ?_) hsub
    /-
      case a.intro.intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hfKU : Set.MapsTo (⇑f) K U
      t : Set (Set Y)
      htT : HasSubset.Subset t T
      htf : t.Finite
      hTU : ∀ (V : Set Y), Membership.mem t V → HasSubset.Subset V U
      hKT : HasSubset.Subset K (Set.preimage (⇑f) t.sUnion)
      s : Set (Set X)
      hsS : HasSubset.Subset s S
      hsf : s.Finite
      hKs : HasSubset.Subset K s.sUnion
      hst : Set.MapsTo (⇑f) s.sUnion t.sUnion
      hsub : HasSubset.Subset (Set.iInter fun L => Set.iInter fun h => setOf fun g = …
      L : Set X
      hL : Membership.mem s L
      ⊢ Membership.mem (iInf fun s => iInf fun h => Filter.principal s) (setOf fun g …
    -/
    refine mem_iInf_of_mem _ <| mem_iInf_of_mem ?_ <| mem_principal_self _
    /-
      case a.intro.intro.intro.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      S : Set (Set X)
      T : Set (Set Y)
      hS₁ : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hT : TopologicalSpace.IsTopologicalBasis T
      hS₂ : ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem T V → Memb …
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hfKU : Set.MapsTo (⇑f) K U
      t : Set (Set Y)
      htT : HasSubset.Subset t T
      htf : t.Finite
      hTU : ∀ (V : Set Y), Membership.mem t V → HasSubset.Subset V U
      hKT : HasSubset.Subset K (Set.preimage (⇑f) t.sUnion)
      s : Set (Set X)
      hsS : HasSubset.Subset s S
      hsf : s.Finite
      hKs : HasSubset.Subset K s.sUnion
      hst : Set.MapsTo (⇑f) s.sUnion t.sUnion
      hsub : HasSubset.Subset (Set.iInter fun L => Set.iInter fun h => setOf fun g = …
      L : Set X
      hL : Membership.mem s L
      ⊢ Membership.mem (setOf fun s => And (Membership.mem s f) (Membership.mem (Set …
    -/
    exact ⟨hst.mono_left (subset_sUnion_of_mem hL), mem_image2_of_mem (hsS hL) ⟨htf, htT⟩⟩
    /-
      🎉 no goals
    -/


/-- A version of `instSecondCountableTopology` with a technical assumption
instead of `[SecondCountableTopology X] [LocallyCompactSpace X]`.
It is here as a reminder of what could be an intermediate goal,
if someone tries to weaken the assumptions in the instance
(e.g., from `[LocallyCompactSpace X]` to `[LocallyCompactPair X Y]` - not sure if it's true). -/
theorem secondCountableTopology [SecondCountableTopology Y]
    (hX : ∃ S : Set (Set X), S.Countable ∧ (∀ K ∈ S, IsCompact K) ∧
      ∀ f : C(X, Y), ∀ V, IsOpen V → ∀ x ∈ f ⁻¹' V, ∃ K ∈ S, K ∈ 𝓝 x ∧ MapsTo f K V) :
    SecondCountableTopology C(X, Y) where
  is_open_generated_countable := by
    /-
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : SecondCountableTopology Y
      hX : Exists fun S => And S.Countable (And (∀ (K : Set X), Membership.mem S K → …
      ⊢ Exists fun b => And b.Countable (Eq ContinuousMap.compactOpen (TopologicalSp …
    -/
    rcases hX with ⟨S, hScount, hScomp, hS⟩
    /-
      case intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : SecondCountableTopology Y
      S : Set (Set X)
      hScount : S.Countable
      hScomp : ∀ (K : Set X), Membership.mem S K → IsCompact K
      hS : ∀ (f : ContinuousMap X Y) (V : Set Y), IsOpen V → ∀ (x : X), Membership.m …
      ⊢ Exists fun b => And b.Countable (Eq ContinuousMap.compactOpen (TopologicalSp …
    -/
    refine ⟨_, ?_, compactOpen_eq_generateFrom (S := S) hScomp (isBasis_countableBasis _) ?_⟩
      /-
        case intro.intro.intro.refine_1
        X : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        inst✝ : SecondCountableTopology Y
        S : Set (Set X)
        hScount : S.Countable
        hScomp : ∀ (K : Set X), Membership.mem S K → IsCompact K
        hS : ∀ (f : ContinuousMap X Y) (V : Set Y), IsOpen V → ∀ (x : X), Membership.m …
        ⊢ (Set.image2 (fun K t => setOf fun f => Set.MapsTo (⇑f) K t.sUnion) S (setOf  …
      -/
    · exact .image2 hScount (countable_setOf_finite_subset (countable_countableBasis Y)) _
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        inst✝ : SecondCountableTopology Y
        S : Set (Set X)
        hScount : S.Countable
        hScomp : ∀ (K : Set X), Membership.mem S K → IsCompact K
        hS : ∀ (f : ContinuousMap X Y) (V : Set Y), IsOpen V → ∀ (x : X), Membership.m …
        ⊢ ∀ (f : ContinuousMap X Y) (x : X) (V : Set Y), Membership.mem (TopologicalSp …
      -/
    · intro f x V hV hx
      /-
        case intro.intro.intro.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        inst✝ : SecondCountableTopology Y
        S : Set (Set X)
        hScount : S.Countable
        hScomp : ∀ (K : Set X), Membership.mem S K → IsCompact K
        hS : ∀ (f : ContinuousMap X Y) (V : Set Y), IsOpen V → ∀ (x : X), Membership.m …
        f : ContinuousMap X Y
        x : X
        V : Set Y
        hV : Membership.mem (TopologicalSpace.countableBasis Y) V
        hx : Membership.mem V (f x)
        ⊢ Exists fun K => And (Membership.mem S K) (And (Membership.mem (nhds x) K) (S …
      -/
      apply hS
      /-
        case intro.intro.intro.refine_2.a
        X : Type u_1
        Y : Type u_2
        inst✝² : TopologicalSpace X
        inst✝¹ : TopologicalSpace Y
        inst✝ : SecondCountableTopology Y
        S : Set (Set X)
        hScount : S.Countable
        hScomp : ∀ (K : Set X), Membership.mem S K → IsCompact K
        hS : ∀ (f : ContinuousMap X Y) (V : Set Y), IsOpen V → ∀ (x : X), Membership.m …
        f : ContinuousMap X Y
        x : X
        V : Set Y
        hV : Membership.mem (TopologicalSpace.countableBasis Y) V
        hx : Membership.mem V (f x)
        ⊢ IsOpen V
      -/
      exacts [isOpen_of_mem_countableBasis hV, hx]
      /-
        🎉 no goals
      -/


instance instSecondCountableTopology [SecondCountableTopology X] [LocallyCompactSpace X]
    [SecondCountableTopology Y] : SecondCountableTopology C(X, Y) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : SecondCountableTopology X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : SecondCountableTopology Y
    ⊢ SecondCountableTopology (ContinuousMap X Y)
  -/
  apply secondCountableTopology
  have (U : countableBasis X) : LocallyCompactSpace U.1 :=
    (isOpen_of_mem_countableBasis U.2).locallyCompactSpace
  /-
    case hX
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : SecondCountableTopology X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : SecondCountableTopology Y
    this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
    ⊢ Exists fun S => And S.Countable (And (∀ (K : Set X), Membership.mem S K → Is …
  -/
  set K := fun U : countableBasis X ↦ CompactExhaustion.choice U.1
  /-
    case hX
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : SecondCountableTopology X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : SecondCountableTopology Y
    this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
    K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
    ⊢ Exists fun S => And S.Countable (And (∀ (K : Set X), Membership.mem S K → Is …
  -/
  use ⋃ U : countableBasis X, Set.range fun n ↦ K U n
  /-
    case h
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : SecondCountableTopology X
    inst✝¹ : LocallyCompactSpace X
    inst✝ : SecondCountableTopology Y
    this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
    K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
    ⊢ And (Set.iUnion fun U => Set.range fun n => Set.image Subtype.val ((K U) n)) …
  -/
  refine ⟨countable_iUnion fun _ ↦ countable_range _, ?_, ?_⟩
    /-
      case h.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      ⊢ ∀ (K_1 : Set X), Membership.mem (Set.iUnion fun U => Set.range fun n => Set. …
    -/
  · simp only [mem_iUnion, mem_range]
    /-
      case h.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      ⊢ ∀ (K_1 : Set X), (Exists fun i => Exists fun y => Eq (Set.image Subtype.val  …
    -/
    rintro K ⟨U, n, rfl⟩
    /-
      case h.refine_1.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      U : ↑(TopologicalSpace.countableBasis X)
      n : Nat
      ⊢ IsCompact (Set.image Subtype.val ((K U) n))
    -/
    exact ((K U).isCompact _).image continuous_subtype_val
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      ⊢ ∀ (f : ContinuousMap X Y) (V : Set Y), IsOpen V → ∀ (x : X), Membership.mem  …
    -/
  · intro f V hVo x hxV
    obtain ⟨U, hU, hxU, hUV⟩ : ∃ U ∈ countableBasis X, x ∈ U ∧ U ⊆ f ⁻¹' V := by
      rw [← (isBasis_countableBasis _).mem_nhds_iff]
      exact (hVo.preimage (map_continuous f)).mem_nhds hxV
    /-
      case h.refine_2.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      f : ContinuousMap X Y
      V : Set Y
      hVo : IsOpen V
      x : X
      hxV : Membership.mem (Set.preimage (⇑f) V) x
      U : Set X
      hU : Membership.mem (TopologicalSpace.countableBasis X) U
      hxU : Membership.mem U x
      hUV : HasSubset.Subset U (Set.preimage (⇑f) V)
      ⊢ Exists fun K_1 => And (Membership.mem (Set.iUnion fun U => Set.range fun n = …
    -/
    lift x to U using hxU
    /-
      case h.refine_2.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      f : ContinuousMap X Y
      V : Set Y
      hVo : IsOpen V
      U : Set X
      hU : Membership.mem (TopologicalSpace.countableBasis X) U
      hUV : HasSubset.Subset U (Set.preimage (⇑f) V)
      x : Subtype fun x => Membership.mem U x
      hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
      ⊢ Exists fun K_1 => And (Membership.mem (Set.iUnion fun U => Set.range fun n = …
    -/
    lift U to countableBasis X using hU
    /-
      case h.refine_2.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      f : ContinuousMap X Y
      V : Set Y
      hVo : IsOpen V
      U : Subtype fun x => Membership.mem (TopologicalSpace.countableBasis X) x
      hUV : HasSubset.Subset (↑U) (Set.preimage (⇑f) V)
      x : Subtype fun x => Membership.mem (↑U) x
      hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
      ⊢ Exists fun K_1 => And (Membership.mem (Set.iUnion fun U => Set.range fun n = …
    -/
    rcases (K U).exists_mem_nhds x with ⟨n, hn⟩
    /-
      case h.refine_2.intro.intro.intro.intro.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : SecondCountableTopology X
      inst✝¹ : LocallyCompactSpace X
      inst✝ : SecondCountableTopology Y
      this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
      K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
      f : ContinuousMap X Y
      V : Set Y
      hVo : IsOpen V
      U : Subtype fun x => Membership.mem (TopologicalSpace.countableBasis X) x
      hUV : HasSubset.Subset (↑U) (Set.preimage (⇑f) V)
      x : Subtype fun x => Membership.mem (↑U) x
      hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
      n : Nat
      hn : Membership.mem (nhds x) ((K U) n)
      ⊢ Exists fun K_1 => And (Membership.mem (Set.iUnion fun U => Set.range fun n = …
    -/
    refine ⟨K U n, mem_iUnion.2 ⟨U, mem_range_self _⟩, ?_, ?_⟩
      /-
        case h.refine_2.intro.intro.intro.intro.intro.intro.refine_1
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X
        inst✝³ : TopologicalSpace Y
        inst✝² : SecondCountableTopology X
        inst✝¹ : LocallyCompactSpace X
        inst✝ : SecondCountableTopology Y
        this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
        K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
        f : ContinuousMap X Y
        V : Set Y
        hVo : IsOpen V
        U : Subtype fun x => Membership.mem (TopologicalSpace.countableBasis X) x
        hUV : HasSubset.Subset (↑U) (Set.preimage (⇑f) V)
        x : Subtype fun x => Membership.mem (↑U) x
        hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
        n : Nat
        hn : Membership.mem (nhds x) ((K U) n)
        ⊢ Membership.mem (nhds ↑x) (Set.image Subtype.val ((K U) n))
      -/
    · rw [← map_nhds_subtype_coe_eq_nhds x.2]
      /-
        case h.refine_2.intro.intro.intro.intro.intro.intro.refine_1
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X
        inst✝³ : TopologicalSpace Y
        inst✝² : SecondCountableTopology X
        inst✝¹ : LocallyCompactSpace X
        inst✝ : SecondCountableTopology Y
        this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
        K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
        f : ContinuousMap X Y
        V : Set Y
        hVo : IsOpen V
        U : Subtype fun x => Membership.mem (TopologicalSpace.countableBasis X) x
        hUV : HasSubset.Subset (↑U) (Set.preimage (⇑f) V)
        x : Subtype fun x => Membership.mem (↑U) x
        hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
        n : Nat
        hn : Membership.mem (nhds x) ((K U) n)
        ⊢ Membership.mem (Filter.map Subtype.val (nhds ⟨↑x, ⋯⟩)) (Set.image Subtype.va …
      -/
      exacts [image_mem_map hn, (isOpen_of_mem_countableBasis U.2).mem_nhds x.2]
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.intro.intro.intro.intro.intro.intro.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X
        inst✝³ : TopologicalSpace Y
        inst✝² : SecondCountableTopology X
        inst✝¹ : LocallyCompactSpace X
        inst✝ : SecondCountableTopology Y
        this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
        K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
        f : ContinuousMap X Y
        V : Set Y
        hVo : IsOpen V
        U : Subtype fun x => Membership.mem (TopologicalSpace.countableBasis X) x
        hUV : HasSubset.Subset (↑U) (Set.preimage (⇑f) V)
        x : Subtype fun x => Membership.mem (↑U) x
        hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
        n : Nat
        hn : Membership.mem (nhds x) ((K U) n)
        ⊢ Set.MapsTo (⇑f) (Set.image Subtype.val ((K U) n)) V
      -/
    · rw [mapsTo_image_iff]
      /-
        case h.refine_2.intro.intro.intro.intro.intro.intro.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : TopologicalSpace X
        inst✝³ : TopologicalSpace Y
        inst✝² : SecondCountableTopology X
        inst✝¹ : LocallyCompactSpace X
        inst✝ : SecondCountableTopology Y
        this : ∀ (U : ↑(TopologicalSpace.countableBasis X)), LocallyCompactSpace ↑↑U
        K : (U : ↑(TopologicalSpace.countableBasis X)) → CompactExhaustion ↑↑U := fun  …
        f : ContinuousMap X Y
        V : Set Y
        hVo : IsOpen V
        U : Subtype fun x => Membership.mem (TopologicalSpace.countableBasis X) x
        hUV : HasSubset.Subset (↑U) (Set.preimage (⇑f) V)
        x : Subtype fun x => Membership.mem (↑U) x
        hxV : Membership.mem (Set.preimage (⇑f) V) ↑x
        n : Nat
        hn : Membership.mem (nhds x) ((K U) n)
        ⊢ Set.MapsTo (Function.comp (⇑f) Subtype.val) ((K U) n) V
      -/
      exact fun y _ ↦ hUV y.2
      /-
        🎉 no goals
      -/


instance instSeparableSpace [SecondCountableTopology X] [LocallyCompactSpace X]
    [SecondCountableTopology Y] : SeparableSpace C(X, Y) :=
  inferInstance


