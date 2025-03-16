/-- The additional data that is necessary in order to obtain a presentation
of the restriction of scalars of a module. -/
abbrev RestrictScalarsData : Type _ :=
  (presB.finsupp presM.G).CokernelData
    (LinearMap.restrictScalars A presM.map)
    (fun (⟨g, g'⟩ : presB.G × presM.R) ↦ presB.var g • Finsupp.single g' (1 : B))


/-- A presentation of the restriction of scalars from `B` to `A` of a `B`-module `M`,
given a presentation of `M` as a `B`-module, a presentation of `B` as an `A`-module,
and an additional data. -/
noncomputable def restrictScalars : Presentation A M :=
  ofExact (g := LinearMap.restrictScalars A presM.π) (presB.finsupp presM.G) data
    presM.exact presM.surjective_π (by
      /-
        B : Type u_1
        inst✝⁸ : Ring B
        M : Type u_2
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module B M
        inst✝⁵ : DecidableEq B
        presM : Module.Presentation B M
        inst✝⁴ : DecidableEq presM.G
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra A B
        inst✝¹ : Module A M
        inst✝ : IsScalarTower A B M
        presB : Module.Presentation A B
        data : presM.RestrictScalarsData presB
        ⊢ Eq (Submodule.span A (Set.range fun x => Module.Presentation.RestrictScalars …
      -/
      ext v
      /-
        case h
        B : Type u_1
        inst✝⁸ : Ring B
        M : Type u_2
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module B M
        inst✝⁵ : DecidableEq B
        presM : Module.Presentation B M
        inst✝⁴ : DecidableEq presM.G
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra A B
        inst✝¹ : Module A M
        inst✝ : IsScalarTower A B M
        presB : Module.Presentation A B
        data : presM.RestrictScalarsData presB
        v : Finsupp presM.R B
        ⊢ Iff (Membership.mem (Submodule.span A (Set.range fun x => Module.Presentatio …
      -/
      dsimp
      /-
        case h
        B : Type u_1
        inst✝⁸ : Ring B
        M : Type u_2
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module B M
        inst✝⁵ : DecidableEq B
        presM : Module.Presentation B M
        inst✝⁴ : DecidableEq presM.G
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra A B
        inst✝¹ : Module A M
        inst✝ : IsScalarTower A B M
        presB : Module.Presentation A B
        data : presM.RestrictScalarsData presB
        v : Finsupp presM.R B
        ⊢ Iff (Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB …
      -/
      simp only [Submodule.mem_top, iff_true]
      /-
        case h
        B : Type u_1
        inst✝⁸ : Ring B
        M : Type u_2
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module B M
        inst✝⁵ : DecidableEq B
        presM : Module.Presentation B M
        inst✝⁴ : DecidableEq presM.G
        A : Type u_3
        inst✝³ : CommRing A
        inst✝² : Algebra A B
        inst✝¹ : Module A M
        inst✝ : IsScalarTower A B M
        presB : Module.Presentation A B
        data : presM.RestrictScalarsData presB
        v : Finsupp presM.R B
        ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
      -/
      apply Finsupp.induction
        /-
          case h.h0
          B : Type u_1
          inst✝⁸ : Ring B
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module B M
          inst✝⁵ : DecidableEq B
          presM : Module.Presentation B M
          inst✝⁴ : DecidableEq presM.G
          A : Type u_3
          inst✝³ : CommRing A
          inst✝² : Algebra A B
          inst✝¹ : Module A M
          inst✝ : IsScalarTower A B M
          presB : Module.Presentation A B
          data : presM.RestrictScalarsData presB
          v : Finsupp presM.R B
          ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          case h.ha
          B : Type u_1
          inst✝⁸ : Ring B
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module B M
          inst✝⁵ : DecidableEq B
          presM : Module.Presentation B M
          inst✝⁴ : DecidableEq presM.G
          A : Type u_3
          inst✝³ : CommRing A
          inst✝² : Algebra A B
          inst✝¹ : Module A M
          inst✝ : IsScalarTower A B M
          presB : Module.Presentation A B
          data : presM.RestrictScalarsData presB
          v : Finsupp presM.R B
          ⊢ ∀ (a : presM.R) (b : B) (f : Finsupp presM.R B), Not (Membership.mem f.suppo …
        -/
      · intro r b w _ _ hw
        /-
          case h.ha
          B : Type u_1
          inst✝⁸ : Ring B
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module B M
          inst✝⁵ : DecidableEq B
          presM : Module.Presentation B M
          inst✝⁴ : DecidableEq presM.G
          A : Type u_3
          inst✝³ : CommRing A
          inst✝² : Algebra A B
          inst✝¹ : Module A M
          inst✝ : IsScalarTower A B M
          presB : Module.Presentation A B
          data : presM.RestrictScalarsData presB
          v : Finsupp presM.R B
          r : presM.R
          b : B
          w : Finsupp presM.R B
          a✝¹ : Not (Membership.mem w.support r)
          a✝ : Ne b 0
          hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
          ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
        -/
        refine Submodule.add_mem _ ?_ hw
        /-
          case h.ha
          B : Type u_1
          inst✝⁸ : Ring B
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module B M
          inst✝⁵ : DecidableEq B
          presM : Module.Presentation B M
          inst✝⁴ : DecidableEq presM.G
          A : Type u_3
          inst✝³ : CommRing A
          inst✝² : Algebra A B
          inst✝¹ : Module A M
          inst✝ : IsScalarTower A B M
          presB : Module.Presentation A B
          data : presM.RestrictScalarsData presB
          v : Finsupp presM.R B
          r : presM.R
          b : B
          w : Finsupp presM.R B
          a✝¹ : Not (Membership.mem w.support r)
          a✝ : Ne b 0
          hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
          ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
        -/
        obtain ⟨β, rfl⟩ := presB.surjective_π b
        /-
          case h.ha.intro
          B : Type u_1
          inst✝⁸ : Ring B
          M : Type u_2
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module B M
          inst✝⁵ : DecidableEq B
          presM : Module.Presentation B M
          inst✝⁴ : DecidableEq presM.G
          A : Type u_3
          inst✝³ : CommRing A
          inst✝² : Algebra A B
          inst✝¹ : Module A M
          inst✝ : IsScalarTower A B M
          presB : Module.Presentation A B
          data : presM.RestrictScalarsData presB
          v : Finsupp presM.R B
          r : presM.R
          w : Finsupp presM.R B
          a✝¹ : Not (Membership.mem w.support r)
          hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
          β : Finsupp presB.G A
          a✝ : Ne (presB.π β) 0
          ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
        -/
        apply Finsupp.induction (p := fun β ↦ Finsupp.single r (presB.π β) ∈ _)
          /-
            case h.ha.intro.h0
            B : Type u_1
            inst✝⁸ : Ring B
            M : Type u_2
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module B M
            inst✝⁵ : DecidableEq B
            presM : Module.Presentation B M
            inst✝⁴ : DecidableEq presM.G
            A : Type u_3
            inst✝³ : CommRing A
            inst✝² : Algebra A B
            inst✝¹ : Module A M
            inst✝ : IsScalarTower A B M
            presB : Module.Presentation A B
            data : presM.RestrictScalarsData presB
            v : Finsupp presM.R B
            r : presM.R
            w : Finsupp presM.R B
            a✝¹ : Not (Membership.mem w.support r)
            hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            β : Finsupp presB.G A
            a✝ : Ne (presB.π β) 0
            ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
          -/
        · simp
          /-
            🎉 no goals
          -/
          /-
            case h.ha.intro.ha
            B : Type u_1
            inst✝⁸ : Ring B
            M : Type u_2
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module B M
            inst✝⁵ : DecidableEq B
            presM : Module.Presentation B M
            inst✝⁴ : DecidableEq presM.G
            A : Type u_3
            inst✝³ : CommRing A
            inst✝² : Algebra A B
            inst✝¹ : Module A M
            inst✝ : IsScalarTower A B M
            presB : Module.Presentation A B
            data : presM.RestrictScalarsData presB
            v : Finsupp presM.R B
            r : presM.R
            w : Finsupp presM.R B
            a✝¹ : Not (Membership.mem w.support r)
            hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            β : Finsupp presB.G A
            a✝ : Ne (presB.π β) 0
            ⊢ ∀ (a : presB.G) (b : A) (f : Finsupp presB.G A), Not (Membership.mem f.suppo …
          -/
        · intro g a f _ _ hf
          /-
            case h.ha.intro.ha
            B : Type u_1
            inst✝⁸ : Ring B
            M : Type u_2
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module B M
            inst✝⁵ : DecidableEq B
            presM : Module.Presentation B M
            inst✝⁴ : DecidableEq presM.G
            A : Type u_3
            inst✝³ : CommRing A
            inst✝² : Algebra A B
            inst✝¹ : Module A M
            inst✝ : IsScalarTower A B M
            presB : Module.Presentation A B
            data : presM.RestrictScalarsData presB
            v : Finsupp presM.R B
            r : presM.R
            w : Finsupp presM.R B
            a✝³ : Not (Membership.mem w.support r)
            hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            β : Finsupp presB.G A
            a✝² : Ne (presB.π β) 0
            g : presB.G
            a : A
            f : Finsupp presB.G A
            a✝¹ : Not (Membership.mem f.support g)
            a✝ : Ne a 0
            hf : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
          -/
          rw [map_add, Finsupp.single_add]
          /-
            case h.ha.intro.ha
            B : Type u_1
            inst✝⁸ : Ring B
            M : Type u_2
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module B M
            inst✝⁵ : DecidableEq B
            presM : Module.Presentation B M
            inst✝⁴ : DecidableEq presM.G
            A : Type u_3
            inst✝³ : CommRing A
            inst✝² : Algebra A B
            inst✝¹ : Module A M
            inst✝ : IsScalarTower A B M
            presB : Module.Presentation A B
            data : presM.RestrictScalarsData presB
            v : Finsupp presM.R B
            r : presM.R
            w : Finsupp presM.R B
            a✝³ : Not (Membership.mem w.support r)
            hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            β : Finsupp presB.G A
            a✝² : Ne (presB.π β) 0
            g : presB.G
            a : A
            f : Finsupp presB.G A
            a✝¹ : Not (Membership.mem f.support g)
            a✝ : Ne a 0
            hf : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
          -/
          refine Submodule.add_mem _ ?_ hf
          rw [← Finsupp.smul_single_one, ← Finsupp.smul_single_one,
            map_smul, Relations.Solution.π_single, smul_assoc]
          /-
            case h.ha.intro.ha
            B : Type u_1
            inst✝⁸ : Ring B
            M : Type u_2
            inst✝⁷ : AddCommGroup M
            inst✝⁶ : Module B M
            inst✝⁵ : DecidableEq B
            presM : Module.Presentation B M
            inst✝⁴ : DecidableEq presM.G
            A : Type u_3
            inst✝³ : CommRing A
            inst✝² : Algebra A B
            inst✝¹ : Module A M
            inst✝ : IsScalarTower A B M
            presB : Module.Presentation A B
            data : presM.RestrictScalarsData presB
            v : Finsupp presM.R B
            r : presM.R
            w : Finsupp presM.R B
            a✝³ : Not (Membership.mem w.support r)
            hw : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            β : Finsupp presB.G A
            a✝² : Ne (presB.π β) 0
            g : presB.G
            a : A
            f : Finsupp presB.G A
            a✝¹ : Not (Membership.mem f.support g)
            a✝ : Ne a 0
            hf : Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.v …
            ⊢ Membership.mem (Submodule.span A (Set.range fun x => HSMul.hSMul (presB.var  …
          -/
          exact Submodule.smul_mem _ _ (Submodule.subset_span ⟨⟨g, r⟩, rfl⟩))
          /-
            🎉 no goals
          -/


